from dataset import LabeledVideoDataModule
from data_pre_processing import Pre_Process
from model import ClassificationModel, AccuracyCallback


import argparse
import pytorch_lightning as pl
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorch_lightning.callbacks import early_stopping


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser(
        description='A 3D CNN Action Recognition model using X3DM model')
    
    # Command-line arguments
    parser.add_argument('--root', type=str, help='Path to the root Video folder')
    parser.add_argument('--train_list', type=str,help='Path to training video file')
    parser.add_argument('--test_list', type=str,help='Path to test video file')
    parser.add_argument('--classes', nargs='+', type=str, help='List of video classes to be extracted (provide space-separated values)')

    parser.add_argument('--frames', type=int, help='The amount of frames to be extracted from each video')
    parser.add_argument('--batch_size', type=int, help='Number of batch size during each epoch')
    parser.add_argument('--num_workers', type=int, help='Number of CPU workers')
    parser.add_argument('--lr', type=float, help='The learning rate for the model')
    parser.add_argument('--device', type=str, help='Accelerator device CPU or GPU')

    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    args = parser.parse_args()

    # Access the parsed arguments
    root = args.root
    train_list = args.train_list
    test_list = args.test_list
    classes = args.classes
    num_classes = len(args.classes)
    frames = args.frames
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    device = args.device


    verbose = args.verbose
    if not os.path.isdir(root):
        raise ValueError("The specified root folder does not exist.")
    
    if not os.path.isfile(train_list) or not os.path.isfile(test_list):
        raise ValueError("One or both of the specified train and test files do not exist.")
    

    #Preprocessing the Paths for dataset
    data_processor = Pre_Process(root_dir=root, train_file=train_list, test_file=test_list, classes=classes)
    train_paths, val_paths, test_paths  = data_processor.extract_video_paths()

    video_dataset = LabeledVideoDataModule(
        train_paths=LabeledVideoPaths(train_paths),
        val_paths=LabeledVideoPaths(val_paths),
        test_paths=LabeledVideoPaths(test_paths),
        frames=frames,batch_size=batch_size,num_workers=num_workers)
    
    print(video_dataset.train_paths.__getitem__(100))
    

    
    model = ClassificationModel(num_classes=num_classes,lr=lr)


    # Callbacks
    accuracy_log = AccuracyCallback()
    earlystopping = early_stopping.EarlyStopping(monitor='V_Acc', patience=5, mode='max', verbose=True)

    # Trainer
    trainer = pl.Trainer(callbacks=[accuracy_log, earlystopping],
                         max_epochs=10, accelerator=device, devices='auto',precision='16')
    
    trainer.fit(model, datamodule=video_dataset)
    trainer.test(model, datamodule=video_dataset)


if __name__ == '__main__':
    main()