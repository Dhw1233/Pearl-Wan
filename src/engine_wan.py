import torch
import time
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from .kvcache_wan import KVCacheModelWAN
from .util_wan import seed_everything, norm_logits, sample, max_fn, simulate_network_delay
from .network_simulator import NetworkSimulator
from .compression import TransmissionCompressor
from .adaptive_window import AdaptiveWindowSelector
from .fallback import FallbackManager

warnings.filterwarnings("ignore")

class PEARLWANEngine:
    """
    PEARL-WAN Engine: Cloud-Edge Collaborative Speculative Decoding
    with Adaptive Window Size for Wide Area Networks.
    
    Architecture:
    - Edge Node: Draft model + network monitor + adaptive decision maker
    - Cloud Node: Target model + verification engine
    - Network Layer: Simulated WAN with configurable RTT/bandwidth/loss
    """
    def __init__(self, args):
        self.args = args
        seed_everything(args.seed)
        
        # Metrics
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        self.wall_times = []
        self.network_times = []
        self.transmission_bytes = []
        
        # Components
        self.network = NetworkSimulator(
            rtt_ms=args.rtt_ms,
            bandwidth_mbps=args.bandwidth_mbps,
            packet_loss_rate=args.packet_loss_rate,
        )
        self.compressor = TransmissionCompressor(
            vocab_size=args.vocab_size,
            enable=args.enable_compression,
            quantize_bits=8,
            top_k_logits=50,
        )
        self.adaptive_window = AdaptiveWindowSelector(
            initial_gamma=args.gamma,
            min_gamma=1,
            max_gamma=16,
            alpha=0.3,
            rtt_ms=args.rtt_ms,
            bandwidth_mbps=args.bandwidth_mbps,
        )
        self.fallback_mgr = FallbackManager(
            threshold_ms=args.fallback_threshold_ms,
            cooldown_rounds=3,
            min_local_tokens=5,
        )
        
        # Load models
        self._load_models()
        self._load_tokenizer()
    
    def _load_models(self):
        print(f"[PEARL-WAN] Loading models...")
        print(f"  Draft (Edge): {self.args.draft_model_path}")
        print(f"  Target (Cloud): {self.args.target_model_path}")
        
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Handle transformers version differences for torch_dtype
        try:
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=self.args.device_edge if self.args.device_edge != "cpu" else None,
            ).eval()
        except TypeError:
            # Older transformers versions may not support torch_dtype param name
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model_path,
                dtype=dtype,
                trust_remote_code=True,
                device_map=self.args.device_edge if self.args.device_edge != "cpu" else None,
            ).eval()
        
        try:
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=self.args.device_cloud if self.args.device_cloud != "cpu" else None,
            ).eval()
        except TypeError:
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model_path,
                dtype=dtype,
                trust_remote_code=True,
                device_map=self.args.device_cloud if self.args.device_cloud != "cpu" else None,
            ).eval()
        
        if self.args.device_edge == "cpu":
            self.draft_model = self.draft_model.to("cpu")
        if self.args.device_cloud == "cpu":
            self.target_model = self.target_model.to("cpu")
        
        print(f"[PEARL-WAN] Models loaded successfully.")
    
    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        print(f"\033[9{color_number}m{content}\033[0m")
    
    def cuda_synchronize(self):
        """Synchronize CUDA if available."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    @torch.no_grad()
    def autoregressive_sampling(self, prefix):
        """Baseline: auto-regressive decoding with target model only."""
        model = self.target_model
        device = model.device
        prefix = prefix.to(device)
        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.args.max_tokens
        
        x = prefix
        past_key_values = None
        start_time = time.time()
        self.cuda_synchronize()
        
        while x.shape[1] < max_tokens:
            if past_key_values is not None:
                last_ids = x[:, -1:]
                outputs = model(last_ids, past_key_values=past_key_values, use_cache=True)
            else:
                outputs = model(x)
            
            last_p = norm_logits(outputs.logits[:, -1, :], self.args.temp, self.args.top_k, self.args.top_p)
            past_key_values = outputs.past_key_values
            idx_next = sample(last_p)
            x = torch.cat((x, idx_next), dim=1)
        
        self.cuda_synchronize()
        total_time = time.time() - start_time
        return x, total_time
    
    @torch.no_grad()
    def speculative_decoding_baseline(self, prefix):
        """Baseline: vanilla speculative decoding (single machine)."""
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        
        draft_cache = KVCacheModelWAN(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        draft_cache.vocab_size = self.args.vocab_size
        target_cache = KVCacheModelWAN(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_cache.vocab_size = self.args.vocab_size
        
        max_tokens = prefix.shape[1] + self.args.max_tokens
        start_time = time.time()
        self.cuda_synchronize()
        
        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            x = draft_cache.generate(prefix.to(draft_device), self.args.gamma)
            _ = target_cache.generate(x.to(target_device), 1)
            self.draft_forward_times += self.args.gamma
            self.target_forward_times += 1
            
            n = prefix_len + self.args.gamma - 1
            for i in range(self.args.gamma):
                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]
                draft_prob = draft_cache._prob_history[:, prefix_len + i - 1, j]
                target_prob = target_cache._prob_history.to(draft_device)[:, prefix_len + i - 1, j]
                if r > target_prob / draft_prob:
                    n = prefix_len + i - 1
                    break
            
            self.num_acc_tokens.append(n - prefix_len + 1)
            prefix = x[:, :n + 1]
            draft_cache.rollback(n + 1)
            
            if n < prefix_len + self.args.gamma - 1:
                t = sample(max_fn(target_cache._prob_history[:, n, :self.args.vocab_size].to(draft_device) 
                                  - draft_cache._prob_history[:, n, :self.args.vocab_size]))
                target_cache.rollback(n + 1)
            else:
                t = sample(target_cache._prob_history[:, -1, :self.args.vocab_size]).to(draft_device)
                target_cache.rollback(n + 2)
            prefix = torch.cat((prefix, t), dim=1)
        
        self.cuda_synchronize()
        total_time = time.time() - start_time
        return prefix, total_time
    
    @torch.no_grad()
    def pearl_wan_decode(self, prefix):
        """
        PEARL-WAN: Cloud-Edge Collaborative Speculative Decoding
        with network simulation, adaptive window, compression, and fallback.
        
        Implements PEARL's pre-verify and post-verify strategies adapted for WAN.
        """
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        
        draft_cache = KVCacheModelWAN(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        draft_cache.vocab_size = self.args.vocab_size
        target_cache = KVCacheModelWAN(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_cache.vocab_size = self.args.vocab_size
        
        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # cur_mode: True = pre-verify mode, False = post-verify mode
        cur_mode = True
        num_acc_token = 0
        prefix = prefix.to(draft_device)
        
        total_start = time.time()
        self.cuda_synchronize()
        
        while prefix.shape[1] < max_tokens:
            round_start = time.time()
            prefix_len = prefix.shape[1]
            
            # --- Fallback Check ---
            if self.args.enable_fallback and self.fallback_mgr.should_fallback():
                local_start = time.time()
                prefix = draft_cache.generate(prefix, 1)
                self.draft_forward_times += 1
                self.fallback_mgr.record_local_token()
                
                # Record local latency (much lower than WAN)
                local_latency_ms = (time.time() - local_start) * 1000
                self.fallback_mgr.record_latency(local_latency_ms * 0.5)
                
                if self.fallback_mgr.should_return_to_cloud():
                    draft_cache.rollback(prefix.shape[1])
                continue
            
            # core logic
            # --- Adaptive Window Size ---
            gamma = self.adaptive_window.get_gamma() if self.args.enable_adaptive_window else self.args.gamma
            
            # --- Edge: Drafting Phase ---
            draft_start = time.time()
            x = draft_cache.generate(prefix, gamma)
            # draft_prob corresponds to positions where draft tokens were generated
            draft_prob = draft_cache._prob_history[:, prefix_len-1:prefix_len+gamma-1, :self.args.vocab_size]
            draft_ids = x[:, prefix_len:prefix_len+gamma]
            self.draft_forward_times += gamma
            draft_time = time.time() - draft_start
            
            # --- Compression & Transmission (Edge -> Cloud) ---
            network_start = time.time()
            compressed_data = self.compressor.compress_logits(draft_prob, draft_ids)
            success, recv_data = self.network.send(compressed_data, simulate_delay=True)
            
            if not success:
                self.fallback_mgr.record_latency(999.0)
                draft_cache.rollback(prefix_len)
                prefix = draft_cache.generate(prefix, 1)
                self.draft_forward_times += 1
                continue
            
            # --- Cloud: Verification Phase ---
            cloud_start = time.time()
            recv_logits, recv_ids = self.compressor.decompress_logits(recv_data, device=target_device)
            
            # Target model runs verification on prefix + draft tokens
            verify_input = torch.cat([prefix.to(target_device), recv_ids.to(target_device)], dim=1)
            _ = target_cache.generate(verify_input, 1)
            target_prob = target_cache._prob_history[:, prefix_len-1:prefix_len+gamma-1, :self.args.vocab_size]
            self.target_forward_times += 1
            cloud_time = time.time() - cloud_start
            
            # --- Network: Return minimal metadata (Cloud -> Edge) ---
            verify_meta = {"type": "verify_result", "gamma": gamma}
            network_meta_start = time.time()
            success, _ = self.network.send(verify_meta, simulate_delay=True)
            network_time = (network_meta_start - network_start) + (time.time() - network_meta_start)
            
            # --- PEARL Verification Logic ---
            if cur_mode:
                # Pre-verify: check first draft token
                first_token = recv_ids[:, 0].to(draft_device)
                torch.manual_seed(self.args.seed + prefix_len)
                r = torch.rand(1, device=draft_device)
                
                draft_p = recv_logits[:, 0, first_token].to(draft_device)
                target_p = target_prob[:, 0, first_token].to(draft_device)
                
                if r > target_p / draft_p:
                    # Reject: resample from target distribution
                    t = sample(max_fn(target_prob[:, 0, :self.args.vocab_size].to(draft_device) 
                                      - recv_logits[:, 0, :self.args.vocab_size].to(draft_device)))
                    prefix = torch.cat((prefix, t), dim=1)
                    
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    draft_cache.rollback(prefix_len)
                    target_cache.rollback(prefix_len)
                    self.adaptive_window.update_acceptance(0, gamma, first_token_rejected=True)
                else:
                    # Accept first token: switch to post-verify, add all gamma candidates
                    cur_mode = False
                    prefix = torch.cat((prefix, recv_ids.to(draft_device)), dim=1)
                    num_acc_token += 1
                    self.adaptive_window.update_acceptance(1, gamma, first_token_rejected=False)
            else:
                # Post-verify: verify all gamma tokens
                n = gamma
                for i in range(gamma):
                    token = recv_ids[:, i].to(draft_device)
                    torch.manual_seed(self.args.seed + prefix_len - gamma + i)
                    r = torch.rand(1, device=draft_device)
                    
                    draft_p = recv_logits[:, i, token].to(draft_device)
                    target_p = target_prob[:, i, token].to(draft_device)
                    
                    if r > target_p / draft_p:
                        n = i
                        break
                
                if n == gamma:
                    # All accepted
                    prefix = torch.cat((prefix.to(draft_device), recv_ids.to(draft_device)), dim=1)
                    num_acc_token += gamma
                    self.adaptive_window.update_acceptance(gamma, gamma, first_token_rejected=False)
                else:
                    # Rejected at position n
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :self.args.vocab_size].to(draft_device) 
                                      - recv_logits[:, n, :self.args.vocab_size].to(draft_device)))
                    prefix = torch.cat((prefix[:, :prefix_len - gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    
                    rollback_pos = prefix_len - gamma + n + 1
                    draft_cache.rollback(rollback_pos)
                    target_cache.rollback(rollback_pos)
                    self.adaptive_window.update_acceptance(n, gamma, first_token_rejected=(n==0))
            
            # Update adaptive window
            if self.args.enable_adaptive_window:
                self.adaptive_window.update_timing(draft_time, cloud_time, network_time, gamma)
                self.adaptive_window.compute_optimal_gamma()
                # Ensure gamma doesn't drop too low in high-latency scenarios
                if self.args.rtt_ms >= 50 and self.adaptive_window.gamma < 3:
                    self.adaptive_window.gamma = 3
            
            # Record latency for fallback
            round_latency_ms = (time.time() - round_start) * 1000
            self.fallback_mgr.record_latency(round_latency_ms)
            self.network_times.append(network_time)
        
        self.cuda_synchronize()
        total_time = time.time() - total_start
        self.wall_times.append(total_time)
        
        return prefix, total_time
    
    def get_stats(self):
        stats = {
            "draft_forward_times": self.draft_forward_times,
            "target_forward_times": self.target_forward_times,
            "mean_accepted_tokens": sum(self.num_acc_tokens) / max(len(self.num_acc_tokens), 1),
            "total_wall_time": sum(self.wall_times),
            "avg_network_time": sum(self.network_times) / max(len(self.network_times), 1) if self.network_times else 0,
            "compression_ratio": self.compressor.get_compression_ratio(),
            "network_stats": self.network.get_stats(),
            "adaptive_window_stats": self.adaptive_window.get_stats(),
            "fallback_stats": self.fallback_mgr.get_stats(),
        }
        return stats
    
    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("PEARL-WAN Statistics")
        print("="*60)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("="*60)
