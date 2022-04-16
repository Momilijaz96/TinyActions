import cv2
import torch
import numpy as np

def load_all_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret = True
    frames = []
    while ret:
        ret, frame = vidcap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    vidcap.release()
    assert len(frames) == frame_count
    frames = torch.from_numpy(np.stack(frames))
    return frames

frames = load_all_frames('video.mp4')
print(frames.shape)