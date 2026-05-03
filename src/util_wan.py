import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time

def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def model_zoo(args):
    """
    Model zoo for PEARL-WAN.
    Supports both local paths and HuggingFace model identifiers.
    """
    vocab_size = {
        "qwen2.5-0.5b": 151936,
        "qwen2.5-0.5b-instruct": 151936,
        "qwen2.5-1.5b": 151936,
        "qwen2.5-1.5b-instruct": 151936,
        "qwen2.5-7b": 152064,
        "qwen2.5-7b-instruct": 152064,
        "deepseek-1.3b": 32256,
        "deepseek-6.7b": 32256,
        "deepseek-33b": 32256,
        "codellama-7b": 32000,
        "codellama-34b": 32000,
        "codellama-70b": 32000,
        "llama-2-7b": 32000,
        "llama-2-70b": 32000,
    }
    
    # Local model paths
    base_dir = "/common/home/hd535/pearl_wan/models"
    zoo = {
        "qwen2.5-0.5b-instruct": f"{base_dir}/qwen2.5-0.5b-instruct",
        "qwen2.5-1.5b-instruct": f"{base_dir}/qwen2.5-1.5b-instruct",
        "qwen2.5-7b-instruct": f"{base_dir}/qwen2.5-7b-instruct",
        "deepseek-1.3b": f"{base_dir}/deepseek-coder-1.3b-base",
        "deepseek-6.7b": f"{base_dir}/deepseek-coder-6.7b-base",
    }
    
    args.vocab_size = vocab_size.get(args.draft_model, 32000)
    args.draft_model_path = zoo.get(args.draft_model, args.draft_model)
    args.target_model_path = zoo.get(args.target_model, args.target_model)

def parse_arguments():
    """Specified arguments for PEARL-WAN."""
    parser = argparse.ArgumentParser(description='PEARL-WAN: Cloud-Edge Collaborative Speculative Decoding')
    
    parser.add_argument('--data_path', type=str, default="/common/home/hd535/pearl_wan/data")
    parser.add_argument('--draft_model', type=str, default="qwen2.5-0.5b-instruct")
    parser.add_argument('--target_model', type=str, default="qwen2.5-1.5b-instruct")
    
    parser.add_argument('--exp_name', '-e', type=str, default="pearl_wan_test", help='folder name for storing results.')
    parser.add_argument('--eval_mode', type=str, default="wan", choices=["wan", "small", "large", "sd"], help='eval mode.')
    parser.add_argument('--num_samples', '-n', type=int, default=1, help='number of samples to evaluate.')
    parser.add_argument('--seed', '-s', type=int, default=1234, help='random seed')
    parser.add_argument('--max_tokens', type=int, default=128, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0.2, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=0, help='top_k for sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for sampling strategy.')
    parser.add_argument('--gamma', type=int, default=4, help='initial window size (guess time).')
    parser.add_argument('--limit', type=int, default=-1, help='limit number of benchmark samples (-1 = all).')
    
    # WAN-specific arguments
    parser.add_argument('--rtt_ms', type=float, default=50.0, help='Network RTT in milliseconds (20-100ms).')
    parser.add_argument('--bandwidth_mbps', type=float, default=100.0, help='Network bandwidth in Mbps.')
    parser.add_argument('--packet_loss_rate', type=float, default=0.0, help='Packet loss rate (0.0-0.05).')
    parser.add_argument('--enable_adaptive_window', action='store_true', default=False, help='Enable adaptive window size (AWAS).')
    parser.add_argument('--enable_compression', action='store_true', default=False, help='Enable transmission compression.')
    parser.add_argument('--enable_fallback', action='store_true', default=False, help='Enable fallback mechanism.')
    parser.add_argument('--fallback_threshold_ms', type=float, default=200.0, help='Latency threshold for fallback (ms).')
    parser.add_argument('--device_edge', type=str, default="cpu", help='Device for edge/draft model.')
    parser.add_argument('--device_cloud', type=str, default="cpu", help='Device for cloud/target model.')
    
    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_zoo(args)
    return args

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter_val = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter_val[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter_mask = cumulative_probs > top_p
        filter_mask[..., 1:] = filter_mask[..., :-1].clone()
        filter_mask[..., 0] = 0
        indices_to_remove = filter_mask.scatter(1, sorted_indices, filter_mask)
        logits[indices_to_remove] = float('-inf')
    return logits

def norm_logits(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
    assert logits.dim() == 2
    if temperature == 0:
        idx = logits.argmax(dim=1)
        new_logits = torch.zeros_like(logits, device=logits.device)
        new_logits[:, idx] = 1
        return new_logits.float()
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs

def sample(probs: torch.Tensor, num_samples: int = 1):
    # Handle edge cases: all zeros, nan, or negative values
    probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
    probs = torch.where(torch.isinf(probs), torch.zeros_like(probs), probs)
    probs = torch.clamp(probs, min=0.0)
    
    # If all probabilities are zero, fall back to uniform distribution
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.shape[-1]
    
    # Ensure probabilities sum to 1
    probs = probs / probs.sum(dim=-1, keepdim=True)
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next

def max_fn(x):
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    # Avoid division by zero
    x_max_sum = torch.where(x_max_sum == 0, torch.ones_like(x_max_sum), x_max_sum)
    return x_max / x_max_sum

def simulate_network_delay(rtt_ms: float, bandwidth_mbps: float, data_bytes: int):
    """
    Simulate network transmission delay.
    delay = RTT/2 + data_size / bandwidth
    """
    propagation_delay = rtt_ms / 2000.0  # convert to seconds (one-way)
    transmission_delay = (data_bytes * 8) / (bandwidth_mbps * 1e6)
    return propagation_delay + transmission_delay
