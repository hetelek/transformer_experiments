from PIL.Image import Image
import data

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter

# tensorboard --logdir=runs
writer = SummaryWriter()


# a = t.Tensor([[1, 1, 1], [2, 2, 2]])
# b = t.Tensor([[1, 1, 1], [2, 2, 2]])
# c = t.cat([a, b], 1)
# print(c.shape)
# print(c)
# exit(0)

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(3, 3, 5, padding=2)
        self.conv2d_2 = nn.Conv2d(3, 3, 5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3072, 100)
        self.linear_2 = nn.Linear(100, 50)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder_1 = ImageEncoder()
    
    def forward(self, batch_video_frames):
        # [batch, frames_per_video, channels_per_frame, frame_width, frame_height]
        assert len(batch_video_frames.shape) == 5
        assert batch_video_frames.shape[2] == 3  # enfore 3 channels

        all_encoded_frames = []
        for i in range(0, len(batch_video_frames)):
            video_frames = batch_video_frames[i]
            encoded_frames = self.image_encoder_1(video_frames)
            all_encoded_frames.append(encoded_frames)

        # output: [batch, num_encoded_frames, frame_encoding]
        output = t.stack(all_encoded_frames)
        assert output.shape[0] == batch_video_frames.shape[0]
        return output



# ve = VideoEncoder()
# i1, _ = data.load_random_batch(100)
# i2, _ = data.load_random_batch(100)
# i = t.stack([i1, i2])
# print('i:', len(i.shape))
# o = ve(i)
# print(o.shape)
# print(o)
# exit(0)

class MetaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(4, 10)  # [speed, steering angle, lateral acceleration, long acceleration]
        self.linear_2 = nn.Linear(10, 50)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

class DecisionMaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.meta_encoder = MetaEncoder()
        self.linear_1 = nn.Linear(100, 200)
        self.linear_2 = nn.Linear(200, 200)
        self.linear_3 = nn.Linear(200, 2) # [target_speed, target_angle]

    def forward(self, images, metas):
        assert images.shape[0] == metas.shape[0]
        encoded_images = self.image_encoder(images)
        encoded_metas = self.meta_encoder(metas)
        x = t.cat([encoded_images, encoded_metas], 1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x

decision_maker = DecisionMaker()
optimizer = t.optim.Adam(decision_maker.parameters(), lr=1e-4)

for epoch in range(0, 1000):
    batch, labels = data.load_random_batch(100)
    batch = batch / 255.

    decision_output = decision_maker(batch, t.randn(100, 4))
    loss = t.sum(t.abs(decision_output - 0))

    print(loss)
    writer.add_scalar("Loss/train", loss, epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        i = data.load_disk_image('test.png')
        i = i.unsqueeze(0)
        decision_output = decision_maker(i, t.randn(1, 4))
        print(decision_output)
        # data.save_image(decision_output)
        # data.save_image(batch[max] * 255., name='img_high_loss.png')
        # data.save_image(batch[min] * 255., name='img_low_loss.png')

writer.close()
