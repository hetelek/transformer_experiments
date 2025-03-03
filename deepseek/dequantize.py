import torch
from safetensors.torch import load_file
import time

def weight_dequant_pytorch(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor in pure PyTorch.
    
    Args:
        x (torch.Tensor): Quantized weight tensor of shape [M, N].
        s (torch.Tensor): Scale tensor of shape [M // block_size, N // block_size].
        block_size (int): Size of the block for dequantization (default: 128).
    
    Returns:
        torch.Tensor: Dequantized weight tensor of shape [M, N].
    """
    # Ensure inputs are contiguous and have correct dimensions
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    
    M, N = x.shape
    M_s, N_s = s.shape
    assert M == M_s * block_size and N == N_s * block_size, "Scale tensor size must match x divided by block_size"
    
    # Reshape x into [M_s, block_size, N_s, block_size]
    x_reshaped = x.view(M_s, block_size, N_s, block_size)
    
    # Reshape s to [M_s, 1, N_s, 1] so it broadcasts across each block
    s_reshaped = s.view(M_s, 1, N_s, 1)
    
    # Perform element-wise multiplication with broadcasting
    y_reshaped = x_reshaped * s_reshaped
    
    # Reshape back to original size [M, N]
    y = y_reshaped.view(M, N)
    
    return y

safetensors_path = f"/Users/hetelek/Desktop/DeepSeek-V3-weights/model-0000{1}-of-000163.safetensors"
state_dict = []
start_time = time.time()
# f = load_file(safetensors_path, device='mps')
f = load_file(safetensors_path, device='cpu')
end_time = time.time()


total_load_time_s = (end_time - start_time)
print(f"total_load_time_ms={total_load_time_s*1000:.2f}ms")
print(f"state_dict.keys()={f.keys()}")

name = 'model.layers.0.mlp.down_proj.weight'
w1 = f[name]
w1 = w1.to(torch.float32)
print(f"name: {w1}")
print(f"dtype: {w1.dtype}")
print(f"shape: {w1.shape}")

name = 'model.layers.0.mlp.down_proj.weight_scale_inv'
wsi1 = f[name]
print(f"name: {wsi1}")
print(f"dtype: {wsi1.dtype}")
print(f"shape: {wsi1.shape}")



# print(w1.shape)

# print(wsi1.shape)

# # Example usage with your tensor sizes
y = weight_dequant_pytorch(w1, wsi1, block_size=128)
print(y.shape)  # Should be [7168, 18432]
print('element_size', y.element_size())  # Should be [7168, 18432]
print(w1[0][0])
print(wsi1[0][0])
print(y[0][0])
end_time = time.time()
total_load_time_s = (end_time - start_time)
print(f"total_load_time_ms={total_load_time_s*1000:.2f}ms")