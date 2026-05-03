import torch
from .util_wan import norm_logits, sample

class KVCacheModelWAN:
    """
    KVCache model wrapper for PEARL-WAN.
    Modified from PEARL's KVCacheModel to support rollback and probability tracking.
    """
    def __init__(self, model: torch.nn.Module, temperature: float = 1, 
                 top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self.vocab_size = None  # Will be set externally
        
        self.forward_count = 0
    
    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(
                    self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p
                )
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # Handle DynamicCache from newer transformers versions
            if hasattr(self._past_key_values, 'get_seq_length'):
                cached_len = self._past_key_values.get_seq_length()
            elif hasattr(self._past_key_values, 'shape'):
                cached_len = self._past_key_values[0][0].shape[2]
            else:
                # Fallback: compute from prob_history
                cached_len = self._prob_history.shape[1]
            
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits[:, :, :self.vocab_size]
            
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
            
            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(
                    not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p
                )
            
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        self.forward_count += 1
        return last_q
    
    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        """Forward the model gamma times."""
        x = prefix
        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:
        output = self._generate_with_kvcache(input_ids, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos: int):
        """Rollback KV cache to end_pos."""
        if hasattr(self._past_key_values, 'crop'):
            self._past_key_values.crop(end_pos)
        else:
            # Manual crop for tuple-based past_key_values
            new_past = []
            for layer_past in self._past_key_values:
                if isinstance(layer_past, tuple):
                    new_layer = tuple(t[:, :, :end_pos, :] for t in layer_past)
                    new_past.append(new_layer)
                else:
                    # Handle other formats
                    new_past.append(layer_past)
            self._past_key_values = tuple(new_past) if isinstance(self._past_key_values, tuple) else new_past
        
        self._prob_history = self._prob_history[:, :end_pos, :]
    
    @torch.no_grad()
    def generate_single(self, input_ids: torch.Tensor):
        """Generate a single token with KV cache."""
        return self.generate(input_ids, 1)
