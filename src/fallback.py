import time

class FallbackManager:
    """
    Fallback mechanism for PEARL-WAN.
    When network latency exceeds threshold, automatically switches to local edge inference.
    """
    def __init__(self, threshold_ms=200.0, cooldown_rounds=3, min_local_tokens=10):
        self.threshold_ms = threshold_ms
        self.cooldown_rounds = cooldown_rounds
        self.min_local_tokens = min_local_tokens
        
        self.fallback_active = False
        self.fallback_round_count = 0
        self.local_tokens_generated = 0
        
        self.latency_history = []
        self.max_history = 10
        
        self.fallback_trigger_count = 0
        self.total_fallback_tokens = 0
    
    def record_latency(self, round_latency_ms: float):
        """Record latency of latest round."""
        self.latency_history.append(round_latency_ms)
        if len(self.latency_history) > self.max_history:
            self.latency_history.pop(0)
    
    def should_fallback(self):
        """Determine if we should activate fallback mode."""
        if self.fallback_active:
            return True
        
        if len(self.latency_history) < 3:
            return False
        
        # Check if recent average latency exceeds threshold
        recent_avg = sum(self.latency_history[-3:]) / 3
        if recent_avg > self.threshold_ms:
            self.fallback_active = True
            self.fallback_round_count = 0
            self.local_tokens_generated = 0
            self.fallback_trigger_count += 1
            return True
        
        return False
    
    def should_return_to_cloud(self):
        """Determine if we should exit fallback mode."""
        if not self.fallback_active:
            return False
        
        self.fallback_round_count += 1
        
        # Stay in fallback for at least cooldown_rounds and min_local_tokens
        if self.fallback_round_count < self.cooldown_rounds:
            return False
        if self.local_tokens_generated < self.min_local_tokens:
            return False
        
        # Check if latency has improved
        if len(self.latency_history) >= 3:
            recent_avg = sum(self.latency_history[-3:]) / 3
            if recent_avg < self.threshold_ms * 0.7:  # 30% below threshold
                self.fallback_active = False
                self.fallback_round_count = 0
                self.local_tokens_generated = 0
                return True
        
        return False
    
    def record_local_token(self):
        """Record generation of a local token during fallback."""
        self.local_tokens_generated += 1
        self.total_fallback_tokens += 1
    
    def is_fallback_active(self):
        return self.fallback_active
    
    def get_stats(self):
        return {
            "fallback_active": self.fallback_active,
            "fallback_trigger_count": self.fallback_trigger_count,
            "total_fallback_tokens": self.total_fallback_tokens,
            "threshold_ms": self.threshold_ms,
        }
