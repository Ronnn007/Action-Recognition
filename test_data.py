from dataset import LabeledVideoDataModule
from data_pre_processing import Pre_Process
from model import ClassificationModel
import pytorch_lightning as pl
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorch_lightning.callbacks import ModelCheckpoint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

classes = ['BlowDryHair', 'Bowling', 'PlayingFlute', 'SoccerPenalty', 'TrampolineJumping']

root_relative_path = 'ActionRecognition/UCF101/ucfTrainTestlist/trainlist02.txt'
root = os.path.abspath(root_relative_path)

train_relative_path = 'ActionRecognition/UCF101/ucfTrainTestlist/trainlist02.txt'
train_p = os.path.abspath(train_relative_path)

test_relative_path = 'ActionRecognition/UCF101/ucfTrainTestlist/testlist02.txt'
test_p = os.path.abspath(test_relative_path)

print(root)
print(train_p)
print(test_p)


data_processor = Pre_Process(root_dir=root, train_file=train_p, test_file=test_p,classes=classes)
train_paths, val_paths, test_paths = data_processor.extract_video_paths()


video_dataset = LabeledVideoDataModule(
    train_paths=LabeledVideoPaths(train_paths),
    val_paths=LabeledVideoPaths(val_paths),
    test_paths=LabeledVideoPaths(test_paths),
    frames=8, batch_size=8, num_workers=6)

model = ClassificationModel.load_from_checkpoint('projects/models/epoch=9-step=590-V_Acc=1.00.ckpt')
test_dataloader = video_dataset.test_dataloader()

trainer = pl.Trainer(accelerator='gpu',devices='auto',precision='16')
trainer.test(model=model, dataloaders=test_dataloader)