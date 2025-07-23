import torch
import torch.nn as nn



class DyPredictor(nn.Module):
    def __init__(self, d_feature=4096, skip_level=10):
        super().__init__()
        self.skip_level = skip_level # 0 skip to 11 skip
        self.predictor = nn.Sequential(
            nn.Linear(d_feature, d_feature // 2),
            nn.ReLU(),
            nn.Linear(d_feature // 2, self.skip_level),
        )
    
    def forward(self, x, ret_logits=False):
        b, n, d = x.shape
        logits  = self.predictor(x)
        
        if ret_logits:
            return logits
        
        probs = torch.softmax(logits, dim=-1)
        return probs 
        