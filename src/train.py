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
        # 3072 = 32 width * 32 height * 3 channels
        self.linear_1 = nn.Linear(3072, 100)
        self.linear_2 = nn.Linear(100, 50)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

class TimeEncoder(nn.Module):
    def __init__(self, encoder_class):
        super().__init__()
        self.encoder_1 = encoder_class()
    
    def forward(self, batch_frames):
        all_encoded_frames = []
        for i in range(0, batch_frames.shape[0]):  # batch_frames.shape[0] or len(batch_frames)?
            frames = batch_frames[i]
            encoded_frames = self.encoder_1(frames)
            all_encoded_frames.append(encoded_frames)

        # output: [batch, num_encoded_frames, frame_encoding]
        output = t.stack(all_encoded_frames)
        assert output.shape[0] == batch_frames.shape[0]
        return output

class MetaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(4, 10)  # [speed, steering angle, lateral acceleration, long acceleration]
        self.linear_2 = nn.Linear(10, 50)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


print('image')
x = TimeEncoder(ImageEncoder)
# in: [num_videos, num_images_per_video, channels_per_image, width, height]
e = x(t.randn(10, 78, 3, 32, 32))
# out: [num_videos, num_images_per_video, frame_encoding]
print(e.shape)

print()
print('meta')

x1 = TimeEncoder(MetaEncoder)
# in: 10 videos, 78 frames each, 4 inputs per frame
e = x1(t.randn(10, 78, 4))
# out: 10 videos, 78 frames each, 50 embedding size
print(e.shape)
exit(0)

class DecisionMaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_encoder = TimeEncoder(ImageEncoder)
        self.meta_encoder = TimeEncoder(MetaEncoder)
        self.linear_1 = nn.Linear(100, 200)
        self.linear_2 = nn.Linear(200, 200)
        self.linear_3 = nn.Linear(200, 2) # [target_speed, target_angle]

    def forward(self, batch_videos, metas):
        assert batch_videos.shape[0] == metas.shape[0]  # check num batches match
        encoded_videos = self.video_encoder(batch_videos)
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
