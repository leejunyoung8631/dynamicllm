import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM



class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # Example output classes
        )
    
    def forward(self, x):
        
        
        return




class LLaMAWithPredictor(nn.Module):
    def __init__(self, config, base_model):
        super().__init__()
        self.base_model = base_model
        
        self.predictor = Predictor(config)
        
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Run the model normally
        outputs = self.base_model.model.embed_tokens(input_ids)
        hidden_states = self.base_model.model.layers[0](outputs, attention_mask=attention_mask)
        print(hidden_states.shape)
        exit()
        
        # Extract features after the first layer
        first_layer_output = hidden_states[0]  # Assuming this is the output from the first layer
        
        # Apply the predictor head (mean pooling across sequence dimension)
        first_layer_pooled = first_layer_output.mean(dim=1)  # Shape: (batch, feature_dim)
        predictor_output = self.predictor(first_layer_pooled)
        
        # Continue with the rest of the model
        for layer in self.base_model.model.layers[1:]:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        final_outputs = self.base_model.lm_head(hidden_states)
        
        return {
            "lm_outputs": final_outputs,
            "predictor_outputs": predictor_output
        }