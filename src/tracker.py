from math import ceil
import json
from t_utils import tensor2list

class Track:
    def __init__(self, video_fps, batch_size) -> None:
        self.storage = {}
        self.video_fps = video_fps
        self.batch_size = batch_size
    
    def update(self, boxes, scores, labels, frame_idx):
        time_sec = ceil(frame_idx / self.video_fps)
        for i in range(len(boxes)):
            self.storage[frame_idx + i] = {
                'boxes': tensor2list(boxes[i]),
                'scores': tensor2list(scores[i]),
                'labels': tensor2list(labels[i]),
                'time_sec': time_sec,
            }

    def save(self, track_path):
        with open(track_path, "w") as data_file:
            json.dump(self.storage, data_file, indent=2)
    def encode(self):
        return json.dumps(self.storage, indent=2)

