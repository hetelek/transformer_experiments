import pickle
import numpy as np
from PIL import Image
import torch as t

_data = None

def get_all_data():
    global _data
    if _data is not None:
        return _data

    file = '/Users/hetelek/Downloads/cifar-10-batches-py/data_batch_1'
    with open(file, 'rb') as fo:
        _data = pickle.load(fo, encoding='bytes')
    return _data

def load_random_image(force_index=None):
    data = get_all_data()

    if force_index:
        i = force_index
    else:
        i = np.random.randint(0, len(data[b'data']))

    label = data[b'filenames'][i].decode('utf-8')
    r = data[b'data'][i][:1024]
    g = data[b'data'][i][1024:1024+1024]
    b = data[b'data'][i][1024+1024:]

    r = t.tensor(r).reshape((32, 32))
    g = t.tensor(g).reshape((32, 32))
    b = t.tensor(b).reshape((32, 32))
    img = t.stack((r, g, b))
    img = img.type(t.float)

    return img, label

def save_image(tensor, name='img_1.png'):
    np_array = tensor.detach().numpy()
    if len(np_array.shape) == 4:
        np_array = np_array.squeeze()
    assert np_array.shape == (3, 32, 32)

    np_array = np_array.astype(np.uint8)
    np_array = np.moveaxis(np_array, 0, 2)

    img = Image.fromarray(np_array)
    img.save(name)

def load_disk_image(name):
    img_data = Image.open(name)
    img_arr = np.array(img_data)
    img_arr = t.tensor(img_arr)
    img_arr = img_arr.transpose(0, 2)
    img_arr = img_arr.transpose(1, 2)
    img_arr = img_arr.type(t.float)
    assert img_arr.shape == (3, 32, 32)
    return img_arr

def load_random_batch(batch_size):
    images = []
    labels = []
    for z in range(0, batch_size):
        i, l = load_random_image(z)
        images.append(i)
        labels.append(l)

    batch = t.stack(images)
    return batch, labels
