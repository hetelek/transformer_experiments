import torch
from safetensors.torch import load_file, save_file
import time

def weight_dequant_2(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes weights by multiplying x with s, sliding s across x.
    
    Args:
        x (torch.Tensor): Quantized weights of shape [M, N]
        s (torch.Tensor): Scale factors of shape [S_M, S_N]
    
    Returns:
        torch.Tensor: Dequantized weights of shape [M, N]
    """
    assert x.dim() == 2 and s.dim() == 2, "Inputs must be 2D tensors"
    M, N = x.shape
    S_M, S_N = s.shape

    # Calculate number of blocks (full tiles of s over x)
    num_blocks_m = M // S_M  # Full blocks along rows
    num_blocks_n = N // S_N  # Full blocks along columns
    rem_m = M % S_M  # Remainder rows
    rem_n = N % S_N  # Remainder columns

    # Reshape x into blocks
    # If there's a remainder, pad x to the next multiple of S_M, S_N
    if rem_m > 0 or rem_n > 0:
        padded_M = (num_blocks_m + (rem_m > 0)) * S_M
        padded_N = (num_blocks_n + (rem_n > 0)) * S_N
        x_padded = torch.zeros(padded_M, padded_N, device=x.device, dtype=x.dtype)
        x_padded[:M, :N] = x
        x = x_padded
        num_blocks_m = padded_M // S_M
        num_blocks_n = padded_N // S_N
    else:
        padded_M, padded_N = M, N

    # Reshape x to [num_blocks_m, S_M, num_blocks_n, S_N]
    x_reshaped = x.view(num_blocks_m, S_M, num_blocks_n, S_N)

    # Reshape s to broadcast over blocks
    s_expanded = s.view(1, S_M, 1, S_N).expand(num_blocks_m, S_M, num_blocks_n, S_N)

    # Element-wise multiplication
    y_reshaped = x_reshaped * s_expanded

    # Reshape back to padded size
    y = y_reshaped.view(padded_M, padded_N)

    # Trim back to original size if padded
    if padded_M != M or padded_N != N:
        y = y[:M, :N]

    return y

safetensors_path = f"/Users/hetelek/Desktop/DeepSeek-V3-weights/model-0000{1}-of-000163.safetensors"
start_time = time.time()
# f = load_file(safetensors_path, device='mps')
f = load_file(safetensors_path, device='cpu')

scale_inv_keys = [k for k in f.keys() if k.endswith('_scale_inv')]
weights_keys = [k.replace('_scale_inv', '') for k in scale_inv_keys]
for i, weights_key in enumerate(weights_keys):
    scale_inv = f[scale_inv_keys[i]]
    scale_inv = scale_inv.to(torch.float16)
    weights = f[weights_key]
    weights = weights.to(torch.float16)

    print(weights_key)
    print(f'weights_shape = {weights.shape}')
    print(f'scale_shape = {scale_inv.shape}')
    print()

    wd_2 = weight_dequant_2(weights, scale_inv)
    f[weights_key] = wd_2
    del f[scale_inv_keys[i]]
end_time = time.time()
print("Time:", end_time - start_time)
time.sleep(10)
# print(f.keys())
# output_path = safetensors_path + '_dequant.safetensors'  # Specify your desired output path
# print(output_path)
# save_file(f, output_path)