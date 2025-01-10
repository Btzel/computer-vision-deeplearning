import os
import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

class Dataset():
    def __init__(self, filelist, filepath, transform=None):
        self.filelist = filelist
        self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        imgpath = os.path.join(self.filepath, self.filelist[index])
        img = Image.open(imgpath).convert('RGB')  # RGB'ye dönüştürüyoruz, siyah-beyazsa hata çıkmaz
        label = 1 if "dog" in imgpath else 0

        if self.transform:
            img = self.transform(img)

        return img, label

# Dosya yollarını tanımla
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(script_dir, 'datasets/catngod/train')
test_dir = os.path.join(script_dir, 'datasets/catngod/test1')

train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

# Görüntü transformasyonları
transformations = transforms.Compose([transforms.Resize((60, 60)), transforms.ToTensor()])

train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)
train, val = random_split(train, [20000, 5000])

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.valid_acc = torchmetrics.Accuracy(task='binary')

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(64 * 5 * 5, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc2 = nn.Linear(128, 1)  # Binary sınıflandırma için tek çıktı nöronu

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        return torch.sigmoid(self.fc2(x))  # Sigmoid çıkış

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits = self(data).squeeze()
        loss = F.binary_cross_entropy(logits, label.float())
        preds = (logits > 0.5).float()
        self.train_acc.update(preds, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_logits = self(val_data).squeeze()
        val_loss = F.binary_cross_entropy(val_logits, val_label.float())
        val_preds = (val_logits > 0.5).float()
        self.valid_acc.update(val_preds, val_label)
        self.log('val_loss', val_loss)

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute())
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log('val_acc', self.valid_acc.compute())
        self.valid_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(train, batch_size=self.hparams.batch_size, shuffle=True,num_workers=7)

    def val_dataloader(self):
        return DataLoader(val, batch_size=self.hparams.batch_size,num_workers=7)


class MyPrintingCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_epoch_end(self, trainer, pl_module):
        print("end of epoch")

    def on_train_end(self, trainer, pl_module):
        print("Training has ended!")

# Model creation
model = LitModel(batch_size=32, learning_rate=1e-3)

# TensorBoard Logger
logger = TensorBoardLogger("tb_logs", name="catngod")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min',
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=os.path.join(script_dir, 'models/lightning_models/catngod'),
    filename='sample-catngod-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

# Trainer setup
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=10,
    logger=logger,
    log_every_n_steps=1,
    callbacks=[early_stop, checkpoint_callback, MyPrintingCallback()]
)

# Start training
trainer.fit(model)

