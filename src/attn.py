import torch
from typing import Optional, Tuple, Any, Dict

from transformers.cache_utils import DynamicCache


# skip_cache
class DyDynamicCache(DynamicCache):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        '''
        buffer = [{ skip_layers : [], feature : [] },
                  { skip_layers : [], feature : [] },
                                  ....
                 ]
        '''
        self.buffer = []
        self.current_position = 0
    
    def add_to_buffer(self, layer_idx, layer_feature):
        if len(self.buffer) == self.current_position:
            self.buffer.append( {"skip_layers": [], "feature": []} )
            self.current_position += 1
        
        self.buffer[-1]["skip_layers"].append(layer_idx)
        self.buffer[-1]["feature"].append(layer_feature)
    
    
    # iter idx from the first and return all the proceeding hidden_states 
    def get_past(self, idx, hidden_state):
        start_idx = None
        for k, entry in enumerate(self.buffer):
            if idx in entry["skip_layers"]:
                start_idx = k
                break
        
        if start_idx is None:
            return hidden_state

        to_cat = []
        for entry in self.buffer[start_idx:]:
            feature = entry["feature"].pop(0) # get & remove
            to_cat.append(feature)
            del entry["skip_layers"][0]
        to_cat.append(hidden_state) # append the last component
    
        self.remove_prev() # check & remove prev     
        
        return torch.cat(to_cat, dim=1)
    
    
    
    # remove previous cahce if it is empty
    def remove_prev(self, ):
        before_len = len(self.buffer)
        self.buffer = [buf for buf in self.buffer if len(buf["skip_layers"]) > 0]
        removed = before_len - len(self.buffer)
        self.current_position = max(0, self.current_position - removed)
        
        
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        
        
        if layer_idx == -100:
            self.key_cache.append(self.key_cache[-1].clone())
            self.value_cache.append(self.value_cache[-1].clone())
            return self.key_cache[-1], self.value_cache[-1]
        
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]



class SkipControl():
    def __init__(self,):
        '''
        token_kwargs = [{ skip_layers : [], feature : [] },
                        { skip_layers : [], feature : [] },
                                        ....
                       ]
        '''
        self.token_kwargs = []
        self.prev_position = -1
        self.current_position = -1
    
    
    def make_input(self, ):
        23123312
    
    
    def remove_prev(self, ):
        32133214