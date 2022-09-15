import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, d_k):
        super().__init__()
        self.d_k = d_k
        self.num_head = num_head
        self.d_model = d_model

        # The output matrices from these linear layers would be 
        # slice into num_head parts to preform multi-headed 
        # attention calculation
        self.w_q = nn.Linear(d_model, d_k*num_head)
        self.w_k = nn.Linear(d_model, d_k*num_head)
        self.w_v = nn.Linear(d_model, d_k*num_head)
        self.final_linear = nn.Linear(d_k*num_head, d_model)
    
    def forward(self, x, attn_mask=None):
        # x: [batch_size, seq_len, d_model]
        # attn_mask (if not None): [batch_size, seq_len]
        batch_size, seq_len, _ = x.shape
        

        # q,k,v: [batch_size, seq_len, num_head*d_k]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # scores, normalized_scores, softmax_scores: [batch_size, seq_len, seq_len]

        # q: [batch_size, seq_len, num_head*d_k] -> [batch_size, num_head, seq_len, d_k]
        q = torch.permute(q, (0, 2, 1)).reshape(batch_size, self.num_head, self.d_k, 
            seq_len).permute(0, 1, 3, 2)
        assert q.shape == (batch_size, self.num_head, seq_len, self.d_k)

        # k: [batch_size, seq_len, num_head*d_k] -> [batch_size, num_head, d_k, seq_len]
        k = torch.permute(k, (0, 2, 1)).reshape(batch_size, self.num_head, self.d_k, 
            seq_len)
        assert k.shape == (batch_size, self.num_head, self.d_k, seq_len)
        
        # scores, normalized_scores, softmax_scores: 
        # [batch_size, num_head, seq_len, seq_len]
        scores = torch.matmul(q, k)
        assert scores.shape == (batch_size, self.num_head, seq_len, seq_len)
    
        normalized_scores = scores/self.d_k
        # Apply attention masks
        if attn_mask is not None:
            attn_mask = torch.where(attn_mask == 0, -float('inf'), attn_mask)
            normalized_scores += attn_mask[:, None, None, :]
            # print(normalized_scores)

        softmax_scores = nn.functional.softmax(normalized_scores, dim=-1)
        #TODO: verify whether the softmax is applied along the correct dimesion

        # v: [batch_size, seq_len, num_head*d_k] -> [batch_size, num_head, seq_len, d_k]
        v = torch.permute(v, (0, 2, 1)).reshape(batch_size, self.num_head, self.d_k, 
            seq_len).permute(0, 1, 3, 2)

        assert v.shape == (batch_size, self.num_head, seq_len, self.d_k)

        # attentions: [batch_size, num_head, seq_len, d_k]
        attentions = torch.matmul(softmax_scores, v)

        # Concat attentions from all the attention heads
        # attentions: [batch_size, num_head, seq_len, d_k] ->
        # [batch_size, seq_len, num_head*d_k]
        attentions = attentions.permute(0,2,1,3).reshape((batch_size, seq_len, self.num_head*self.d_k))
        assert attentions.shape == (batch_size, seq_len, self.num_head*self.d_k)

        # Final linear layer: 
        #[batch_size, seq_len, num_head*d_k] -> [batch_size, seq_len, d_model]
        outputs = self.final_linear(attentions)
        assert outputs.shape == (batch_size, seq_len, self.d_model)

        return outputs
        # return {
        #     'k': k,
        #     'q': q,
        #     'v': v,
        #     'softmax_scores': softmax_scores,
        #     'attentions': attentions,
        #     'outputs': outputs
        # }
        # return outputs
            