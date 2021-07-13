import av
import data
import torch as t
import random


def load_video(video_file):
    video = av.open(video_file)
    video_frames = []
    for frame in video.decode():
        if not isinstance(frame, av.VideoFrame):
            continue
        frame_data = t.tensor(frame.to_rgb().to_ndarray())
        frame_data = t.movedim(frame_data, (0, 1, 2), (1, 2, 0))
        video_frames.append(frame_data)
    return t.stack(video_frames)


def stitch_images(images):
    width, height = images[0][0].shape[1], images[0][0].shape[2]
    stitched = t.zeros((3, width*len(images), height*len(images[0])))
    for r, image_rows in enumerate(images):
        for c, image in enumerate(image_rows):
            stitched[:, r*width:(r+1)*width, c*height:(c+1)*height] = image
    return stitched

video_frames = load_video('clip3.mp4')
final = stitch_images([
    [random.choice(video_frames), random.choice(video_frames)],
    [random.choice(video_frames), random.choice(video_frames)],
    [random.choice(video_frames)]
])
data.save_image(final)
