import torch
# from safetensors.torch import load_file, save_file
# import time

def weight_dequant_2(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes weights by multiplying x with s, sliding s across x.
    
    Args:
        x (torch.Tensor): Quantized weights of shape [M, N]
        s (torch.Tensor): Scale factors of shape [S_M, S_N]
    
    Returns:
        torch.Tensor: Dequantized weights of shape [M, N]
    """
    # Emulate Triton weight_dequant kernel behavior using 2D block tiling
    assert x.dim() == 2 and s.dim() == 2, "Inputs must be 2D tensors"
    # Promote to float32 for accurate computation (Triton kernel uses float32)
    x = x.to(torch.float32)
    s = s.to(torch.float32)
    M, N = x.shape
    # Number of blocks along columns inferred from s
    num_blocks_cols = s.shape[1]
    # Derive block size (assumes square blocks) from N and number of column blocks
    block_size = N // num_blocks_cols
    # Number of blocks along rows
    num_blocks_rows = (M + block_size - 1) // block_size
    # Flatten scale tensor (row-major) and allocate output
    s_flat = s.flatten()
    y = torch.empty_like(x)
    # Loop over block rows and columns
    for i in range(num_blocks_rows):
        r0 = i * block_size
        r1 = min(r0 + block_size, M)
        for j in range(num_blocks_cols):
            c0 = j * block_size
            c1 = min(c0 + block_size, N)
            idx = i * num_blocks_cols + j
            scale = s_flat[idx]
            y[r0:r1, c0:c1] = x[r0:r1, c0:c1] * scale
    return y


"""
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
"""