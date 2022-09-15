import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__() 
        self.dropout = nn.Dropout(p = dropout_rate)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        dims = torch.arange(0, d_model, device=x.device)
        positions = torch.arange(0, seq_len, device=x.device)
        encoding = torch.tensor([[self.pos_encode(d, p, d_model) for d in dims] for p in positions], device=x.device)
        encoding = self.dropout(encoding)

        return self.dropout(x + encoding)

        # x: [batch, seq_len, d_model]
        
    def pos_encode(self, dim, pos, d_model):
        if dim%2==0:
            return torch.sin(pos/(torch.pow(10000, dim/d_model)))
        else:
            return torch.cos(pos/(torch.pow(10000, dim/d_model)))