import pytorch_lightning as pl
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
#from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
import itertools




from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomRotation,
    RandomAffine,
    RandomHorizontalFlip,
    RandomPosterize,
)

class LimitDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos

class LabeledVideoDataModule(pl.LightningDataModule):
    def __init__(self, train_paths, val_paths, test_paths, frames, batch_size, num_workers, **kwargs):
        super().__init__()
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.frames = frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        for k,v in kwargs.items():
            setattr(self, k, v)

        self.train_transforms = Compose([
            ApplyTransformToKey(
                key='video',
                transform=Compose([
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomRotation((-60, 40)),
                    RandomHorizontalFlip(p=0.9),
                    RandomPosterize(bits=2),
                ])
            ),
        ])

        self.val_transforms = Compose([
            ApplyTransformToKey(
                key='video',
                transform=Compose([
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(256),
                    CenterCrop(244)
                ])
            ),
        ])

    def train_dataloader(self):
        train_dataset = LimitDataset(
            LabeledVideoDataset(
            labeled_video_paths=self.train_paths,
            clip_sampler=make_clip_sampler('random', self.frames),
            decode_audio=False,
            transform=self.train_transforms,
            )
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        val_dataset = LimitDataset(
            LabeledVideoDataset(
            labeled_video_paths=self.val_paths,
            clip_sampler=make_clip_sampler('random', self.frames),
            decode_audio=False,
            transform=self.val_transforms,
            )
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        test_dataset = LimitDataset(
            LabeledVideoDataset(
            labeled_video_paths=self.test_paths,
            clip_sampler=make_clip_sampler('random', self.frames),
            decode_audio=False,
            transform=self.val_transforms,
            )
        )
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    


    