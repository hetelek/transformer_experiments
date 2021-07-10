import pickle
import time
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

def save_image(tensor, name='img_1.png'):
    np_array = tensor.detach().numpy()
    if len(np_array.shape) == 4:
        np_array = np_array.squeeze()
    assert np_array.shape == (3, 32, 32)

    np_array = np_array.astype(np.uint8)
    np_array = np.moveaxis(np_array, 0, 2)
    
    img = Image.fromarray(np_array)
    img.save(name)

def load_random_image():
    i = np.random.randint(0, len(dict[b'data']))
    r = dict[b'data'][i][:1024]
    g = dict[b'data'][i][1024:1024+1024]
    b = dict[b'data'][i][1024+1024:]

    r = torch.tensor(r).reshape((32, 32))
    g = torch.tensor(g).reshape((32, 32))
    b = torch.tensor(b).reshape((32, 32))
    img = torch.stack((r, g, b))

    return img

# for _ in range(0, 100):
#     i, _ = load_random_image()
#     time.sleep(1)

i = load_random_image()

batch = i.clone().detach().type(torch.float) # torch.tensor(i, dtype=torch.float)
batch = torch.unsqueeze(batch, 0)

c = nn.Conv2d(3, 6, 5, padding=(2, 2))(batch)
c = nn.Conv2d(6, 10, 5, padding=(2, 2))(c)
c = nn.Conv2d(10, 50, 5, padding=(2, 2))(c)
c = nn.Conv2d(50, 30, 5, padding=(2, 2))(c)
c = nn.Conv2d(30, 3, 5, padding=(2, 2))(c)

save_image(c, name='final.png')

