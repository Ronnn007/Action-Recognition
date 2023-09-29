import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import pickle
#from pytorchvideo.models import create_res_basic_head


class ClassificationModel(pl.LightningModule):
    '''
        Pytorch lightning Module for NN model.
        X3DM model is used (pre-trained on kinetics 400 dataset)
        Optimizer & learning rate schedular are defined.
        Test step is also used for testing after training is completed on unseen data.

    '''

    def __init__(self, num_classes, lr, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.lr = lr
        self.model = self.create_x3dm(num_classes)

    def create_x3dm(self, num_classes):
        x3d = torch.hub.load('facebookresearch/pytorchvideo','x3d_m', pretrained=True)
        x3d.blocks[-1].proj = nn.Linear(2048, num_classes)
        return x3d

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        videos, labels = batch['video'], batch['label']
        outputs = self(videos)

        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).sum().item() / len(labels)

        self.log("T_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, batch_size=batch_idx)
        self.log("T_Acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, batch_size=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch['video'], batch['label']
        outputs = self(videos)

        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).sum().item() / len(labels)

        self.log("V_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, batch_size=batch_idx)
        self.log("V_Acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, batch_size=batch_idx)

        return loss

    def test_step(self, batch, batch_idx):
        videos, labels = batch['video'], batch['label']
        outputs = self(videos)

        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).sum().item() / len(labels)

        metrics = {"Test accuracy": acc, "Test_loss": loss}
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=0.001)
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.ASGD(
            self.parameters(), lr=self.hparams.lr, weight_decay= 0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1, verbose=True)
        # return [optimizer], [scheduler]
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
    

class AccuracyCallback(Callback):
    '''
        This callback is used for accuracy and losses graph creation.
        Rather than using the default loggers based approach from pytorch lightning this callback bypasses that,
        and enables 3rd party library such as matplotlib graph creations
    '''

    def __init__(self):
        super().__init__()
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect the training accuracy from the trainer
        train_acc = trainer.callback_metrics.get('T_Acc')
        train_loss = trainer.callback_metrics.get('T_loss')
        self.train_accuracies.append(train_acc.item())
        self.train_losses.append(train_loss.item())

        #print(f"Train Accuracies: {self.train_accuracies}")
        #print(f"Train Accuracies Length: {len(self.train_accuracies)}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect the validation accuracy from the trainer
        val_acc = trainer.callback_metrics.get('V_Acc')
        val_loss = trainer.callback_metrics.get('V_loss')
        self.val_accuracies.append(val_acc.item())
        self.val_losses.append(val_loss.item())

        #print(f"Validation Accuracies: {self.val_accuracies}")
        #print(f"Validation Accuracies Length: {len(self.val_accuracies)}")

    def on_train_end(self, trainer, pl_module):

        if len(self.val_accuracies) > len(self.train_accuracies):
            self.val_accuracies.pop()

        if len(self.val_losses) > len(self.train_losses):
            self.val_losses.pop()

        # Plotting the training and validation accuracies using matplotlib
        epochs = len(self.train_accuracies)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), self.train_accuracies,
                 label='Train Accuracy')
        plt.plot(range(1, epochs + 1), self.val_accuracies,
                 label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train and Validation Accuracies')
        plt.legend()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), self.val_losses,
                 label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.show()
        plt.savefig('home/ec22362/projects/projects/loss_plot.png')
        
        #Retriving the model performance data for later analysis
        training_metrics = {'train_accuracies': self.train_accuracies,
                             'val_accuracies': self.val_accuracies,
                             'train_losses': self.train_losses,
                             'val_losses': self.val_losses}
        
        with open('training_metrics.pkl', 'wb') as f:
            pickle.dump(training_metrics, f)

