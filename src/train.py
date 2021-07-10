import data

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter

# tensorboard --logdir=runs
writer = SummaryWriter()

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(3, 3, 5, padding=2)
        self.conv2d_2 = nn.Conv2d(3, 3, 5, padding=2)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        return x

ae = AutoEncoder()
optimizer = t.optim.Adam(ae.parameters(), lr=1e-4)

for epoch in range(0, 1000):
    batch, labels = data.load_random_batch(100)
    batch = batch / 255.

    encoder_output = ae(batch)

    loss = t.sum(t.abs(batch - encoder_output))
    print(loss)
    writer.add_scalar("Loss/train", loss, epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        for i in range(0, encoder_output.shape[0]):
            data.save_image(encoder_output[i] * 255., name='imgs/{}.png'.format(i))

        i = data.load_disk_image('test.png')
        i = i.unsqueeze(0)
        encoder_output = ae(i)

        data.save_image(encoder_output)
        # data.save_image(batch[max] * 255., name='img_high_loss.png')
        # data.save_image(batch[min] * 255., name='img_low_loss.png')

writer.close()
