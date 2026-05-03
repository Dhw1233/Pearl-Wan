import time
import random
import pickle
import torch

class NetworkSimulator:
    """
    Simulates Wide Area Network (WAN) conditions between edge and cloud.
    Configurable RTT, bandwidth, and packet loss rate.
    """
    def __init__(self, rtt_ms=50.0, bandwidth_mbps=100.0, packet_loss_rate=0.0, jitter_ms=5.0):
        self.rtt_ms = rtt_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.packet_loss_rate = packet_loss_rate
        self.jitter_ms = jitter_ms
        self.total_bytes_sent = 0
        self.total_delay_injected = 0.0
        self.packet_drop_count = 0
        self.packet_count = 0
        
    def _simulate_one_way_delay(self, data_bytes: int):
        """Calculate one-way network delay."""
        base_propagation = self.rtt_ms / 2000.0  # one-way propagation (seconds)
        jitter = random.uniform(-self.jitter_ms, self.jitter_ms) / 1000.0
        transmission_delay = (data_bytes * 8) / (self.bandwidth_mbps * 1e6)
        return base_propagation + jitter + transmission_delay
    
    def send(self, data, simulate_delay=True):
        """
        Simulate sending data over the network.
        Returns (success: bool, received_data: any).
        If packet is lost, returns (False, None).
        """
        self.packet_count += 1
        
        # Serialize to calculate size
        if isinstance(data, torch.Tensor):
            data_bytes = data.element_size() * data.nelement()
        else:
            data_bytes = len(pickle.dumps(data))
        
        self.total_bytes_sent += data_bytes
        
        # Simulate packet loss
        if random.random() < self.packet_loss_rate:
            self.packet_drop_count += 1
            return False, None
        
        # Simulate network delay
        if simulate_delay:
            delay = self._simulate_one_way_delay(data_bytes)
            time.sleep(delay)
            self.total_delay_injected += delay
        
        return True, data
    
    def get_stats(self):
        return {
            "total_bytes_sent": self.total_bytes_sent,
            "total_delay_injected_sec": self.total_delay_injected,
            "packet_count": self.packet_count,
            "packet_drop_count": self.packet_drop_count,
            "packet_loss_rate_actual": self.packet_drop_count / max(self.packet_count, 1),
        }
    
    def reset_stats(self):
        self.total_bytes_sent = 0
        self.total_delay_injected = 0.0
        self.packet_drop_count = 0
        self.packet_count = 0
