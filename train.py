import pickle
import time
import numpy as np
from PIL import Image

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


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

def load_random_image(force_index=None):
    i = np.random.randint(0, len(dict[b'data'])) if not force_index else force_index
    label = dict[b'filenames'][i].decode('utf-8')
    r = dict[b'data'][i][:1024]
    g = dict[b'data'][i][1024:1024+1024]
    b = dict[b'data'][i][1024+1024:]

    r = torch.tensor(r).reshape((32, 32))
    g = torch.tensor(g).reshape((32, 32))
    b = torch.tensor(b).reshape((32, 32))
    img = torch.stack((r, g, b))
    img = img.type(torch.float)

    return img, label

def load_disk_image(name):
    img_data = Image.open(name)
    img_arr = np.array(img_data)
    img_arr = torch.tensor(img_arr)
    img_arr = img_arr.transpose(0, 2)
    img_arr = img_arr.transpose(1, 2)
    img_arr = img_arr.type(torch.float)
    assert img_arr.shape == (3, 32, 32)
    return img_arr

# for _ in range(0, 100):
#     i, _ = load_random_image()
#     time.sleep(1)

def load_batch(batch_size):
    images = []
    labels = []
    for z in range(0, batch_size):
        i, l = load_random_image(z)
        i = i / 255.
        images.append(i)
        labels.append(l)

    batch = torch.stack(images)
    return batch, labels

# batch = batch.type(torch.float) # torch.tensor(i, dtype=torch.float)
# batch = torch.unsqueeze(batch, 0)

c = nn.Conv2d(3, 3, 5, padding=(2, 2))
# c2 = nn.Conv2d(6, 10, 5, padding=(2, 2))
# c3 = nn.Conv2d(10, 50, 5, padding=(2, 2))
# c4 = nn.Conv2d(50, 30, 5, padding=(2, 2))
c_full = nn.Linear(32, 200)
c_full2 = nn.Linear(200, 200)
c_full3 = nn.Linear(200, 32)
# c_full = nn.Linear(32, 960)
c5 = nn.Conv2d(3, 3, 5, padding=(2, 2))
d = nn.Dropout(p=0.4)

def forward(x):
    o = c_full(x) #c(x)
    o = c_full2(o)
    o = c_full3(o)
    # o = c2(o)
    # o = d(c3(o))
    # o = c4(o)
    # print(o.shape)
    return o #c5(d(c_full(o)))

# all = [d, c, c2, c3, c4, c5]
all = [d, c, c5, c_full, c_full2, c_full3]
params = []
for a in all:
    params += list(a.parameters())

# optimizer = torch.optim.SGD(params, lr=1e-6, momentum=0.9)
optimizer = torch.optim.Adam(params, lr=1e-2)

for epoch in range(0, 10000):
    batch, labels = load_batch(1000)
    o = forward(batch)
    loss = torch.abs(o - batch)
    # print(loss.shape)
    # max = torch.argmax(loss)
    # print(max)
    # print(loss)
    loss2 = torch.sum(loss, 1)
    loss2 = torch.sum(loss2, 1)
    loss2 = torch.sum(loss2, 1)
    max = torch.argmax(loss2)
    min = torch.argmin(loss2)
    # print(labels[x])
    # print()

    loss = torch.sum(loss)
    print(loss)
    writer.add_scalar("Loss/train", loss, epoch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        for i in range(0, o.shape[0]):
            save_image(o[i] * 255., name='imgs/{}.png'.format(i))

        i = load_disk_image('test.png')
        i = i.unsqueeze(0)
        o = forward(i)
        save_image(o)

        save_image(batch[max] * 255., name='img_high_loss.png')
        save_image(batch[min] * 255., name='img_low_loss.png')


writer.close()