import time
import torch

class AdaptiveWindowSelector:
    """
    Adaptive Window Size Algorithm (AWAS) for PEARL-WAN.
    Dynamically selects optimal gamma (window size) based on:
    - Recent acceptance rate
    - Network conditions (RTT, bandwidth)
    - Model computation speed
    """
    def __init__(self, initial_gamma=4, min_gamma=1, max_gamma=16, 
                 alpha=0.3, rtt_ms=50.0, bandwidth_mbps=100.0):
        self.gamma = initial_gamma
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.alpha = alpha  # EMA smoothing factor
        
        self.rtt_ms = rtt_ms
        self.bandwidth_mbps = bandwidth_mbps
        
        # Historical metrics
        self.acceptance_rate_ema = 0.5
        self.recent_acceptances = []
        self.rejected_first_token_count = 0
        self.total_rounds = 0
        
        # Timing estimates
        self.avg_draft_time_per_token = 0.0
        self.avg_target_verify_time = 0.0
        self.avg_network_time = 0.0
        
        # Adaptive parameters
        self.gamma_history = []
    
    def update_acceptance(self, num_accepted: int, gamma: int, first_token_rejected: bool = False):
        """Update acceptance statistics after each round."""
        self.total_rounds += 1
        if gamma > 0:
            rate = num_accepted / gamma
        else:
            rate = 0.0
        
        self.recent_acceptances.append(num_accepted)
        if len(self.recent_acceptances) > 20:
            self.recent_acceptances.pop(0)
        
        # Update EMA of acceptance rate
        self.acceptance_rate_ema = self.alpha * rate + (1 - self.alpha) * self.acceptance_rate_ema
        
        if first_token_rejected:
            self.rejected_first_token_count += 1
        
        self.gamma_history.append(self.gamma)
    
    def update_timing(self, draft_time: float, target_time: float, network_time: float, gamma: int):
        """Update timing estimates."""
        if gamma > 0:
            draft_time_per_tok = draft_time / gamma
        else:
            draft_time_per_tok = draft_time
        
        self.avg_draft_time_per_token = 0.3 * draft_time_per_tok + 0.7 * self.avg_draft_time_per_token
        self.avg_target_verify_time = 0.3 * target_time + 0.7 * self.avg_target_verify_time
        self.avg_network_time = 0.3 * network_time + 0.7 * self.avg_network_time
    
    def compute_optimal_gamma(self):
        """
        Compute optimal gamma based on cost model.
        
        Cost per token approximation:
        - If acceptance rate is high, larger gamma is better (amortizes network cost)
        - If acceptance rate is low, smaller gamma reduces wasted computation
        - Network RTT dominates: higher RTT -> need larger gamma
        
        We model expected tokens per round:
        E[accepted] = gamma * acceptance_rate_ema
        
        Cost per round:
        cost = draft_time + network_time + target_verify_time
        
        Effective time per accepted token:
        time_per_token = cost / E[accepted]
        
        We want to minimize time_per_token by choosing gamma.
        Assuming:
        - draft_time ≈ gamma * t_draft
        - target_verify_time ≈ t_target (mostly fixed for small gamma)
        - network_time ≈ RTT + gamma * t_transmit
        
        Then:
        time_per_token = (gamma * t_draft + RTT + gamma * t_transmit + t_target) / (gamma * p)
                     = (t_draft + t_transmit) / p + (RTT + t_target) / (gamma * p)
        
        To minimize, we want larger gamma when RTT is large.
        But we also need to consider that acceptance rate drops with gamma.
        
        Simplified heuristic:
        gamma ∝ RTT / (t_draft + t_transmit) * acceptance_rate_ema
        """
        if self.avg_draft_time_per_token <= 0:
            # No timing info yet, use heuristic based on RTT
            base_gamma = max(1, int(self.rtt_ms / 20))
        else:
            # Heuristic: scale gamma with network-to-compute ratio
            compute_time = self.avg_draft_time_per_token + (self.avg_network_time / max(self.gamma, 1))
            network_compute_ratio = self.avg_network_time / max(compute_time, 1e-6)
            
            base_gamma = int(self.gamma * (0.5 + 0.5 * network_compute_ratio) * (0.5 + 0.5 * self.acceptance_rate_ema))
        
        # Adjust based on first-token rejection rate
        if self.total_rounds > 5:
            first_token_reject_rate = self.rejected_first_token_count / self.total_rounds
            if first_token_reject_rate > 0.5:
                # High first-token rejection -> use pre-verify strategy, smaller gamma
                base_gamma = max(self.min_gamma, base_gamma - 2)
            elif first_token_reject_rate < 0.2 and self.acceptance_rate_ema > 0.7:
                # Low rejection, high acceptance -> can afford larger gamma
                base_gamma = min(self.max_gamma, base_gamma + 2)
        
        # Clamp to valid range
        new_gamma = max(self.min_gamma, min(self.max_gamma, base_gamma))
        
        # Smooth transition: don't change too abruptly
        if abs(new_gamma - self.gamma) > 2:
            new_gamma = self.gamma + (1 if new_gamma > self.gamma else -1) * 2
        
        self.gamma = new_gamma
        return self.gamma
    
    def get_gamma(self):
        return self.gamma
    
    def get_stats(self):
        return {
            "current_gamma": self.gamma,
            "acceptance_rate_ema": self.acceptance_rate_ema,
            "total_rounds": self.total_rounds,
            "avg_draft_time_per_token": self.avg_draft_time_per_token,
            "avg_target_verify_time": self.avg_target_verify_time,
            "avg_network_time": self.avg_network_time,
        }
