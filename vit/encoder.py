import torch
from torch import nn
from vit.multiheaded_attentions import MultiHeadAttention
from vit.positional_encoding import PositionalEncoding

class FeedForwardNetword(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.l1 = nn.Linear(d_model, d_model)
        self.l2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.l2(nn.functional.relu(self.l1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_k, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_head, d_k)
        self.ffn = FeedForwardNetword(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p = dropout_rate)
    
    def forward(self, x, attn_mask):
        # x: [batch_size, seq_len, d_model]
        # attentions_outputs = self.self_attention(x)
        outputs = self.dropout(self.self_attention(x, attn_mask=attn_mask))
        # outputs = attentions_outputs['outputs']
        outputs = self.layer_norm(x + outputs)
        outputs = self.layer_norm(outputs + self.dropout(self.ffn(outputs)))
        return outputs

class EncoderStack(nn.Module):
    def __init__(self, num_layer, d_model, num_head, d_k, dropout_rate):
        super().__init__()
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_k
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_head, d_k, dropout_rate) for i in range(num_layer)])

    def forward(self, x, attn_mask):
        for i, layer, in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layer, num_head, d_k, dropout_rate, max_len=512):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layer = num_layer
        self.num_head = num_head
        self.d_k = d_k

        # An embedding layer that transforms tokens to model embeddings
        # See: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(dropout_rate, d_model, max_len)
        self.layers = EncoderStack(self.num_layer, self.d_model, self.num_head, self.d_k, dropout_rate)

    def forward(self, x, attn_mask):
        # x: [batch_size, seq_len]
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.layers(x, attn_mask)
        return x

class TransformerEncoderClassifer(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d_model, 
        num_layer, 
        num_head, 
        d_k, 
        dropout_rate,
        num_class,
        max_len
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size, 
            d_model=d_model, 
            num_layer=num_layer, 
            num_head=num_head, 
            d_k=d_k, 
            dropout_rate=dropout_rate,
            max_len=max_len
        )
        self.pre_classifier = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_class)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, attn_mask=None):
        # x: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.encoder(x, attn_mask=attn_mask)
        # Here we use the hidden_state of the first token as the input for
        # classification
        # x: [batch_size, seq_len, d_model] -> pooled_x: [batch_size, d,model]
        pooled_x = x[:, 0]
        pooled_x = self.pre_classifier(pooled_x)
        pooled_x = nn.ReLU()(pooled_x)
        pooled_x = self.dropout(pooled_x)
        outputs = self.classifier(pooled_x)
        return outputs
