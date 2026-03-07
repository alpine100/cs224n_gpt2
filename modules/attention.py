import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### YOUR CODE HERE
    #check if nn.Module training attr is set before applying dropout
    dropout_p = self.dropout.p if self.training else 0
    #key positions (s)
    S = key.size(-2)
    # Create causal mask on the same device and with the same dtype as attention_mask
    causal_mask = torch.triu(torch.full((S, S), -10000.0, device=attention_mask.device, dtype=attention_mask.dtype), diagonal=1)
    
    # longformer mask 
    window_size = 512 # cite
    window_mask = torch.triu(torch.full((S, S), -10000.0, device=attention_mask.device, dtype=attention_mask.dtype), diagonal=-window_size)
    
    # combine masks together
    longformer_mask = causal_mask + window_mask

    # longformer global attention
    resulting_mask = longformer_mask + attention_mask

    # attention_mask = causal mask (s,s) + padding(bs,h,query_positions,key_positions(s))
    #                  add padding to respective cols in mask
    # resulting_mask = causal_mask + attention_mask

    attention_output = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=resulting_mask, dropout_p=dropout_p)
    
    # convert from (bs, num_heads, seq_len, head_dim) to (bs, seq_len, hidden_size)
    result = rearrange(attention_output, 'b h t d -> b t (h d)')

    return result


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
