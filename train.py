
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


file = '/Users/hetelek/Downloads/cifar-10-batches-py/data_batch_1'
with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
# dict['pontoon_s_000643']
# dict[b'data']

def save_image(tensor):
    np_array = tensor.detach().numpy()
    np_array = np_array.astype(np.uint8)
    np_array = np_array.squeeze()
    np_array = np.moveaxis(np_array, 0, 2)
    print(np_array.dtype)
    
    img = Image.fromarray(np_array)
    img.save('img_1.png')

def load_random_image():
    i = np.random.randint(0, len(dict[b'data']))
    r = dict[b'data'][i][:1024]
    g = dict[b'data'][i][1024:1024+1024]
    b = dict[b'data'][i][1024+1024:]

    r = np.array(r).reshape((32, 32))
    g = np.array(g).reshape((32, 32))
    b = np.array(b).reshape((32, 32))

    array = np.stack((r, g, b))
    array = np.moveaxis(array, 0, -1)
    print(array.shape)
    img = Image.fromarray(array)
    img.save('img_orig.png')

    return array, dict[b'filenames'][i]

i, _ = load_random_image()

batch = torch.tensor(i, dtype=torch.float)
batch = torch.unsqueeze(batch, 0)
batch = torch.transpose(batch, 1, 2)
batch = torch.transpose(batch, 1, 3)

save_image(batch)

c = nn.Conv2d(3, 6, 5, padding=(2, 2))(batch)
c = nn.Conv2d(6, 10, 5, padding=(2, 2))(c)
c = nn.Conv2d(10, 50, 5, padding=(2, 2))(c)
c = nn.Conv2d(50, 30, 5, padding=(2, 2))(c)
c = nn.Conv2d(30, 3, 5, padding=(2, 2))(c)

# c5 = c3 + fully_connect(c)

# save_image(batch)
# t = save_image(c)