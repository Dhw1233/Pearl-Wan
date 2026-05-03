import torch
import numpy as np

class TransmissionCompressor:
    """
    Lightweight transmission protocol for WAN environment.
    Compresses token logits and candidate tokens before transmission.
    """
    def __init__(self, vocab_size: int, enable=True, quantize_bits=8, top_k_logits=50):
        self.vocab_size = vocab_size
        self.enable = enable
        self.quantize_bits = quantize_bits
        self.top_k_logits = top_k_logits
        
        self.original_bytes = 0
        self.compressed_bytes = 0
    
    def compress_logits(self, logits: torch.Tensor, draft_ids: torch.Tensor):
        """
        Compress draft model output for transmission.
        
        Args:
            logits: Tensor of shape (batch, seq_len, vocab_size)
            draft_ids: Tensor of shape (batch, seq_len) - candidate tokens
        
        Returns:
            compressed_data: dict containing compressed information
        """
        if not self.enable:
            return {"type": "raw", "logits": logits, "draft_ids": draft_ids}
        
        batch_size, seq_len, vocab = logits.shape
        self.original_bytes += logits.element_size() * logits.nelement()
        self.original_bytes += draft_ids.element_size() * draft_ids.nelement()
        
        # Strategy 1: Quantize logits from float32/bfloat16 to int8
        # Using per-token min-max quantization
        quantized_logits = None
        scales = []
        if self.quantize_bits == 8:
            # Flatten per token: (batch * seq_len, vocab)
            flat_logits = logits.reshape(-1, vocab)
            min_vals = flat_logits.min(dim=-1, keepdim=True).values
            max_vals = flat_logits.max(dim=-1, keepdim=True).values
            scales = (max_vals - min_vals) / 255.0
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            quantized = ((flat_logits - min_vals) / scales).to(torch.uint8)
            quantized_logits = quantized.reshape(batch_size, seq_len, vocab)
            scales = scales.reshape(batch_size, seq_len, 1)
            min_vals = min_vals.reshape(batch_size, seq_len, 1)
        
        # Strategy 2: Only keep top-k logits + sparse representation for rest
        # This significantly reduces bandwidth when top_k << vocab_size
        if self.top_k_logits > 0 and self.top_k_logits < vocab:
            top_k_vals, top_k_indices = torch.topk(logits, self.top_k_logits, dim=-1)
            # Send only top-k values and indices
            compressed_data = {
                "type": "topk_quantized",
                "top_k_vals": top_k_vals.cpu(),  # keep as float for accuracy on top-k
                "top_k_indices": top_k_indices.cpu().to(torch.int32),
                "draft_ids": draft_ids.cpu().to(torch.int32),
                "shape": (batch_size, seq_len, vocab),
            }
        else:
            compressed_data = {
                "type": "quantized",
                "quantized_logits": quantized_logits.cpu(),
                "scales": scales.cpu(),
                "min_vals": min_vals.cpu(),
                "draft_ids": draft_ids.cpu().to(torch.int32),
                "shape": (batch_size, seq_len, vocab),
            }
        
        # Estimate compressed size
        if compressed_data["type"] == "topk_quantized":
            est_bytes = (compressed_data["top_k_vals"].element_size() * compressed_data["top_k_vals"].nelement() +
                        compressed_data["top_k_indices"].element_size() * compressed_data["top_k_indices"].nelement() +
                        compressed_data["draft_ids"].element_size() * compressed_data["draft_ids"].nelement())
        else:
            est_bytes = (compressed_data["quantized_logits"].element_size() * compressed_data["quantized_logits"].nelement() +
                        compressed_data["scales"].element_size() * compressed_data["scales"].nelement() +
                        compressed_data["min_vals"].element_size() * compressed_data["min_vals"].nelement() +
                        compressed_data["draft_ids"].element_size() * compressed_data["draft_ids"].nelement())
        
        self.compressed_bytes += est_bytes
        return compressed_data
    
    def decompress_logits(self, compressed_data, device="cpu"):
        """
        Decompress data received from edge node.
        """
        if compressed_data["type"] == "raw":
            return compressed_data["logits"].to(device), compressed_data["draft_ids"].to(device)
        
        batch_size, seq_len, vocab = compressed_data["shape"]
        draft_ids = compressed_data["draft_ids"].to(device)
        
        if compressed_data["type"] == "topk_quantized":
            # Reconstruct full logits with top-k values, rest as -inf
            top_k_vals = compressed_data["top_k_vals"].to(device)
            top_k_indices = compressed_data["top_k_indices"].to(device)
            
            logits = torch.full((batch_size, seq_len, vocab), float('-inf'), 
                               dtype=top_k_vals.dtype, device=device)
            logits.scatter_(-1, top_k_indices, top_k_vals)
            return logits, draft_ids
        
        elif compressed_data["type"] == "quantized":
            quantized = compressed_data["quantized_logits"].to(device)
            scales = compressed_data["scales"].to(device)
            min_vals = compressed_data["min_vals"].to(device)
            
            logits = quantized.float() * scales + min_vals
            return logits, draft_ids
        
        else:
            raise ValueError(f"Unknown compression type: {compressed_data['type']}")
    
    def get_compression_ratio(self):
        if self.original_bytes == 0:
            return 1.0
        ratio = self.original_bytes / max(self.compressed_bytes, 1)
        return ratio
    
    def reset_stats(self):
        self.original_bytes = 0
        self.compressed_bytes = 0
