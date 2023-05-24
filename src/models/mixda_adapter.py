import torch
from torch import nn
import math

# Implement example reweighting
class WeightTensor(nn.Module):
    def __init__(self, example_num, batch_size):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(batch_size)) for _ in range((example_num - example_num % batch_size) // batch_size)])
        if example_num % batch_size != 0:
            self.weights.append(nn.Parameter(torch.ones(example_num % batch_size)))
    def forward(self, batch_id):
        return self.weights[batch_id].softmax(dim=0)

# Main class, Mixture-of-Domain adapters
class MixtureOfDomainAdapter(nn.Module):
    def __init__(self, config, down_scale=None, input_size=None, in_feature=None, mid_feature=None, out_feature=None):
        super().__init__()
        self.config = config
        adapter_down_scale = down_scale if down_scale is not None else config.adapter_down_scale
        self.down_sample = int(config.intermediate_size // adapter_down_scale)
        self.input_size = input_size if input_size else config.intermediate_size
        self.output_size = config.hidden_size
        if in_feature is not None and mid_feature is not None and out_feature is not None:
            self.input_size, self.down_sample, self.output_size = in_feature, mid_feature, out_feature
        self.adapter_down = nn.Sequential(
            nn.Linear(self.input_size, self.down_sample),
            nn.GELU(),
        )
        self.adapter_up = nn.Linear(self.down_sample, self.output_size)
        
        # initialize weights
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.adapter_down[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down[0].bias)
            nn.init.zeros_(self.adapter_up.bias)
            nn.init.zeros_(self.adapter_up.weight)
    def forward(self, x):
        down = self.adapter_down(x)
        up = self.adapter_up(down)
        output = up
        return output
    
# Attention Layer for Transformer outputs + mixture-of-domain adapters
class MixDAAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        kdim = config.hidden_size
        vdim = config.hidden_size
        qdim = config.hidden_size
        num_attention_heads = config.attn_n_head
        dropout = config.attn_dropout
        self.attn = nn.MultiheadAttention(qdim, num_attention_heads, dropout=dropout, kdim=kdim, vdim=vdim,
                                          batch_first=True)
    def forward(self, adapter_output, transformer_output, query):
        if len(transformer_output.shape) == 2:
            combined_k = torch.stack([transformer_output, adapter_output], dim=1)
            query = query.unsqueeze(1)
            return self.attn(query, combined_k, combined_k)[0]
        else:
            orig_shape = adapter_output.shape
            query = query.view(-1, query.shape[-1])
            query = query.unsqueeze(1)
            adapter_output = adapter_output.view(-1, adapter_output.shape[-1])
            transformer_output = transformer_output.view(-1, transformer_output.shape[-1])
            combined_k = torch.stack([transformer_output, adapter_output], dim=1)
            result = self.attn(query, combined_k, combined_k)[0]
            return result.view(orig_shape)
