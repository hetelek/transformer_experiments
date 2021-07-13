import av
import data
import torch as t


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

frame_data = load_video('clip.m4v')
frame_data2 = load_video('clip2.m4v')
print(frame_data[0].shape)
print(frame_data2[0].shape)
final = t.cat([frame_data[0], frame_data2[100]], 1)
final = t.cat([final, frame_data2[300]], 1)
print(final.shape)
data.save_image(final)
