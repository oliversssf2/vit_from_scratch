import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dropout_rate, d_model, max_len):
        super().__init__() 
        self.dropout = nn.Dropout(p = dropout_rate)
    
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encodings = torch.zeros(max_len, 1, d_model)
        pos_encodings[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 0, 1::2] = torch.cos(position * div_term)
        pos_encodings = pos_encodings.squeeze(1)
        self.register_buffer('pos_encodings', pos_encodings)

        print(self.pos_encodings.shape)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # dims = torch.arange(0, d_model, device=x.device)
        # positions = torch.arange(0, seq_len, device=x.device)
        # encoding = torch.tensor([[self.pos_encode(d, p, d_model) for d in dims] for p in positions], device=x.device)
        encoding = self.pos_encodings[:seq_len]
        encoding = self.dropout(encoding)

        return self.dropout(x + encoding)

        # x: [batch, seq_len, d_model]
        
    # def pos_encode(self, dim, pos, d_model):
    #     return self.pos_encodings[pos]
        # if dim%2==0:
        #     return torch.sin(pos/(torch.pow(10000, dim/d_model)))
        # else:
        #     return torch.cos(pos/(torch.pow(10000, dim/d_model)))