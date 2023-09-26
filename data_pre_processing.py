import os
from sklearn.model_selection import train_test_split
#from dataset import LabeledVideoDataModule
#from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
import cv2
from PIL import Image

class Pre_Process():
    def __init__(self, root_dir, train_file, test_file, classes):
       self.root_dir = root_dir
       self.train_split = train_file
       self.test_split = test_file
       self.classes = classes
    
    def extract_video_paths(self):
        train_paths, val_paths = [],[]

        with open(self.train_split, 'r') as f:
           train_videos = f.read().splitlines()
           for train_video in train_videos:
               paths, _ = train_video.split(' ')
               labels = paths.split('/')[0]
               if self.classes and labels not in self.classes:
                    continue
               train_paths.append((os.path.join(self.root_dir, 'UCF-101', paths), self.classes.index(labels)))

        with open(self.test_split, 'r') as f:
            test_videos = f.read().splitlines()
            for test_video in test_videos:
                paths = test_video.split(' ')[0]
                labels = paths.split('/')[0]
                if self.classes and labels not in self.classes:
                    continue
                val_paths.append((os.path.join(self.root_dir,'UCF-101',paths),self.classes.index(labels)))
        
        label2id = {class_label: class_id for class_label, class_id in enumerate(self.classes)}

        split_ratio = 0.97
        train_paths, test_paths = train_test_split(train_paths, train_size=split_ratio)

        for video_path, label in train_paths:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")


        print('Training videos:', len(train_paths))
        print('Validation videos:', len(val_paths))
        print('Testing videos:', len(test_paths))

        return train_paths, val_paths, test_paths

    
    def extract_frame_paths(self,root_f):
        paths_labels = {'train': [], 'val': [], 'test': []}

        for split in os.listdir(root_f):
            split_path = os.path.join(root_f, split)
            if os.path.isdir(split_path):
                # (class folders) Iterate through the classes within the split

                for class_name in os.listdir(split_path):
                    class_path = os.path.join(split_path, class_name)
                    class_label = int(class_name)

                    if os.path.isdir(class_path):
                        # (Video Folders) Iterate through the video folders within the class
                        for video_name in os.listdir(class_path):
                            video_path = os.path.join(class_path, video_name)

                            if os.path.basename(os.path.dirname(class_path)) == 'train_frames':
                                paths_labels['train'].append(
                                    (video_path, class_label))
                            if os.path.basename(os.path.dirname(class_path)) == 'test_frames':
                                paths_labels['test'].append(
                                    (video_path, class_label))
                            if os.path.basename(os.path.dirname(class_path)) == 'val_frames':
                                paths_labels['val'].append(
                                    (video_path, class_label))
        return paths_labels
    
    def convert_to_frames(self,video_paths, output_root):
        for video_path, label in video_paths:
            frames_dir = os.path.join(output_root, str(label), os.path.splitext(os.path.basename(video_path))[0])
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir, exist_ok=True)
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    image = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image.save(os.path.join(
                        frames_dir, f"frame_{frame_count:04d}.jpg"))
                cap.release()
