import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        """
        LoRA Linear layer that adds trainable low-rank matrices A and B.
        The original weight is frozen, and only A and B are updated.
        """
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # Original weight is initialized and then frozen.
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight.requires_grad = False  # Freeze original weight
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.bias.requires_grad = False  # Freeze bias
        else:
            self.register_parameter('bias', None)

        # LoRA parameters (trainable low-rank factors)
        if r > 0:
            self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def forward(self, x):
        if self.r > 0:
            # Compute the LoRA update: B @ A, scaled by alpha, and add it to the frozen weight.
            weight_eff = self.weight + self.alpha * (self.B @ self.A)
        else:
            weight_eff = self.weight
        return F.linear(x, weight_eff, self.bias)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        # Using standard linear layers inside the residual block
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.norm(out)
        return x + out

class LoRAAudioToImageMapper(nn.Module):
    def __init__(self, input_dim=512, output_dim=768, hidden_dim=1024, r=4, num_res_blocks=2, dropout=0.1):
        """
        A LoRA-adapted mapping function that transforms a 512-dim audio embedding
        into a 768-dim conditioning vector for Stable Diffusion.
        """
        super(LoRAAudioToImageMapper, self).__init__()
        # Initial projection: uses LoRALinear so that only LoRA factors are trained.
        self.initial_linear = LoRALinear(input_dim, hidden_dim, r=r)
        self.initial_relu = nn.ReLU()
        self.initial_norm = nn.LayerNorm(hidden_dim)
        
        # Residual blocks (using standard layers; you can also adapt these if desired)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)])
        
        # Final projection: again using LoRALinear
        self.final_linear = LoRALinear(hidden_dim, output_dim, r=r)
        
    def forward(self, x):
        out = self.initial_linear(x)
        out = self.initial_relu(out)
        out = self.initial_norm(out)
        out = self.res_blocks(out)
        out = self.final_linear(out)
        return out

# Example usage (for testing):
if __name__ == "__main__":
    mapper = LoRAAudioToImageMapper()
    dummy_audio_emb = torch.randn(1, 512)
    mapped_embedding = mapper(dummy_audio_emb)
    print("Mapped embedding shape:", mapped_embedding.shape)  # Expected: [1, 768]
