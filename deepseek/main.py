import torch
from safetensors.torch import load_file
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"

'''
/Users/hetelek/Desktop/:
Run 1 (safetensors files 00001 - 00008)
Total loaded: 35423620480
Loading time: 41326.16ms
Load speed: 817.46 MB/s

Run 2 (safetensors files 00001 - 00008)
Total loaded: 35423620480
Loading time: 36594.41ms
Load speed: 923.16 MB/s

--------

/Volumes/T7/DeepSeek-V3-671B/:
Run 1 (safetensors files 00001 - 00008):
Total loaded: 35423620480
Loading time: 75228.20ms
Load speed: 449.07 MB/s

Run 2 (safetensors files 00001 - 00008):
Total loaded: 35423620480
Loading time: 75395.60ms
Load speed: 448.07 MB/s
'''

# from deepseek: 5G file
all_weights_size = 0
safetensors_paths = [f"/Volumes/T7/DeepSeek-V3-671B/model-0000{i}-of-000163.safetensors" for i in range(1, 9)]
file_to_weights_size = {}
for safetensors_path in safetensors_paths:
    with open(safetensors_path, 'rb') as f:
        header_size_bytes = f.read(8)
        header_size = int.from_bytes(header_size_bytes, byteorder='little', signed=False)
        f.seek(0, 2)
        weights_size = f.tell() - header_size
        file_to_weights_size[safetensors_path] = weights_size
    all_weights_size += weights_size
    print(f"Total size: {all_weights_size}. Current file: {weights_size} bytes")
print("Device:", device)
print(file_to_weights_size)

state_dict = []
total_loaded = 0
start_time = time.time()
for safetensors_path in safetensors_paths:
    state_dict.append(load_file(safetensors_path, device=device))
    total_loaded += file_to_weights_size[safetensors_path]
    # print('.')
    # print(state_dict[-1].keys())
    # exit(0)
    # print(f"Total loaded: {total_loaded}")
    # print(f'loaded {safetensors_path}')
end_time = time.time()
total_load_time_s = (end_time - start_time)

print(f"Total loaded: {total_loaded}")
print(f"Loading time: {total_load_time_s*1000:.2f}ms")
load_speed_per_sec = all_weights_size / total_load_time_s
print(f"Load speed: {load_speed_per_sec/1024/1024:.2f} MB/s")
