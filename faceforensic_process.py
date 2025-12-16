import cv2
import os
from pathlib import Path
from tqdm import tqdm
import random


class VideoFrameExtractor:
    """
    Extract frames from videos at fixed intervals for deepfake detection training.
    """

    def __init__(self, dataset_root, output_root, frame_interval=30, val_split=0.2):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.frame_interval = frame_interval
        self.val_split = val_split

        self.folders = {
            'real': {
                'source': self.dataset_root / 'original',
                'train_output': self.output_root / 'train' / 'real',
                'val_output': self.output_root / 'val' / 'real'
            },
            'fake': {
                'source': self.dataset_root / 'DeepFakeDetection',
                'train_output': self.output_root / 'train' / 'fake',
                'val_output': self.output_root / 'val' / 'fake'
            }
        }

        for info in self.folders.values():
            info['train_output'].mkdir(parents=True, exist_ok=True)
            info['val_output'].mkdir(parents=True, exist_ok=True)

    def extract_frames_from_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return 0

        frame_count = 0
        extracted = 0
        video_name = video_path.stem

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_interval == 0:
                out_path = output_dir / f"{video_name}_frame_{extracted:04d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted += 1

            frame_count += 1

        cap.release()
        return extracted

    def process_folder(self, label):
        info = self.folders[label]
        src = info['source']

        if not src.exists():
            print(f"Warning: {src} does not exist, skipping.")
            return

        videos = list(src.glob("*.mp4"))
        if not videos:
            print(f"No videos found in {src}")
            return

        random.shuffle(videos)
        split = int(len(videos) * (1 - self.val_split))
        train_videos = videos[:split]
        val_videos = videos[split:]

        print(f"\nProcessing {label}: {len(videos)} videos")

        for v in tqdm(train_videos, desc=f"{label} train"):
            self.extract_frames_from_video(v, info['train_output'])

        for v in tqdm(val_videos, desc=f"{label} val"):
            self.extract_frames_from_video(v, info['val_output'])

    def process_all(self):
        random.seed(42)

        print("Starting frame extraction")
        print(f"Frame interval: {self.frame_interval}")
        print(f"Output root: {self.output_root}")

        self.process_folder("real")
        self.process_folder("fake")

        print("\nDone.")
        print(f"Dataset created at {self.output_root}")


def main():
    DATASET_ROOT = Path("all_datasets/FaceForensics").resolve()
    OUTPUT_ROOT = Path("all_datasets/postprocess").resolve()

    FRAME_INTERVAL = 30
    VAL_SPLIT = 0.2

    extractor = VideoFrameExtractor(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        frame_interval=FRAME_INTERVAL,
        val_split=VAL_SPLIT
    )

    extractor.process_all()

if __name__ == "__main__":
    main()
