import cv2
import os
from pathlib import Path
from tqdm import tqdm

class VideoFrameExtractor:
    """
    Extract frames from videos at fixed intervals for deepfake detection training.
    """
    
    def __init__(self, dataset_root, output_root, frame_interval=30, val_split=0.2):
        """
        Initialize the frame extractor.
        
        Args:
            dataset_root (str): Root directory of the Kaggle dataset
            output_root (str): Directory where extracted frames will be saved
            frame_interval (int): Extract one frame every N frames (default: 30, ~1 fps for 30fps video)
            val_split (float): Fraction of data to use for validation (default: 0.2)
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.frame_interval = frame_interval
        self.val_split = val_split
        
        # Define source and destination folders
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
        
        # Create output directories
        for folder_info in self.folders.values():
            folder_info['train_output'].mkdir(parents=True, exist_ok=True)
            folder_info['val_output'].mkdir(parents=True, exist_ok=True)
    
    def extract_frames_from_video(self, video_path, output_dir, label):
        """
        Extract frames from a single video file.
        
        Args:
            video_path (Path): Path to the video file
            output_dir (Path): Directory to save extracted frames
            label (str): Label for the frames ('real' or 'fake')
        
        Returns:
            int: Number of frames extracted
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0
        
        frame_count = 0
        extracted_count = 0
        video_name = video_path.stem
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % self.frame_interval == 0:
                frame_filename = f"{video_name}_frame_{extracted_count:04d}.jpg"
                frame_path = output_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        return extracted_count
    
    def process_folder(self, label):
        """
        Process all videos in a folder (real or fake) and split into train/val.
        
        Args:
            label (str): 'real' or 'fake'
        """
        import random
        
        folder_info = self.folders[label]
        source_dir = folder_info['source']
        train_output = folder_info['train_output']
        val_output = folder_info['val_output']
        
        if not source_dir.exists():
            print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
            return
        
        # Get all MP4 files
        video_files = list(source_dir.glob('*.mp4'))
        
        if not video_files:
            print(f"No MP4 files found in {source_dir}")
            return
        
        # Shuffle and split videos into train/val
        random.shuffle(video_files)
        split_idx = int(len(video_files) * (1 - self.val_split))
        train_videos = video_files[:split_idx]
        val_videos = video_files[split_idx:]
        
        print(f"\nProcessing {len(video_files)} {label} videos from {source_dir}")
        print(f"Train videos: {len(train_videos)} | Val videos: {len(val_videos)}")
        
        # Process training videos
        total_train_frames = 0
        for video_path in tqdm(train_videos, desc=f"Extracting {label} train frames"):
            frames_extracted = self.extract_frames_from_video(video_path, train_output, label)
            total_train_frames += frames_extracted
        
        # Process validation videos
        total_val_frames = 0
        for video_path in tqdm(val_videos, desc=f"Extracting {label} val frames"):
            frames_extracted = self.extract_frames_from_video(video_path, val_output, label)
            total_val_frames += frames_extracted
        
        print(f"Extracted {total_train_frames} train frames and {total_val_frames} val frames from {label} videos")
    
    def process_all(self):
        """
        Process all videos from both real and fake folders.
        """
        import random
        random.seed(42)  # For reproducible train/val splits
        
        print(f"Starting frame extraction...")
        print(f"Frame interval: 1 frame every {self.frame_interval} frames")
        print(f"Train/Val split: {int((1-self.val_split)*100)}% / {int(self.val_split*100)}%")
        print(f"Output directory: {self.output_root}")
        
        # Process real videos
        self.process_folder('real')
        
        # Process fake videos
        self.process_folder('fake')
        
        print("\nFrame extraction complete!")
        print(f"Dataset structure created at: {self.output_root}")
        print(f"├── train/")
        print(f"│   ├── real/  ({len(list(self.folders['real']['train_output'].glob('*.jpg')))} images)")
        print(f"│   └── fake/  ({len(list(self.folders['fake']['train_output'].glob('*.jpg')))} images)")
        print(f"└── val/")
        print(f"    ├── real/  ({len(list(self.folders['real']['val_output'].glob('*.jpg')))} images)")
        print(f"    └── fake/  ({len(list(self.folders['fake']['val_output'].glob('*.jpg')))} images)")


def main():
    """
    Main function to run the frame extraction.
    """
    # Configuration
    DATASET_ROOT = "/hdd/zlim135/Git/dct-cryptonets/all_dataset/FaceForensic/preprocessed"
    OUTPUT_ROOT = "/hdd/zlim135/Git/dct-cryptonets/all_dataset/FaceForensic/postprocessed"
    FRAME_INTERVAL = 30  # Extract 1 frame every 30 frames (~1 fps for 30fps video)
    VAL_SPLIT = 0.2      # 20% of videos for validation
    
    # Create extractor and process videos
    extractor = VideoFrameExtractor(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        frame_interval=FRAME_INTERVAL,
        val_split=VAL_SPLIT
    )
    
    extractor.process_all()


if __name__ == "__main__":
    main()