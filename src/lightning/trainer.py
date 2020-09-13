import logging
from argparse import ArgumentParser

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CometLogger
# Local imports
from data import CIFAR10DataModule
from models import CIFARTenLitModel

logging.info(pl.__version__)

# set seed
pl.seed_everything(42)

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=128)

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

# Transforms
tfms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])


# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)
# comet_logger = CometLogger(
#     api_key='r3QI6mx4KaB3v0VMFwt6bcf33',
#     workspace='katnoria',  # Optional
#     save_dir='.',  # Optional
#     project_name='cf10-pl',  # Optional
# )

# trainer = pl.Trainer(
#     fast_dev_run=False, 
#     gpus=1, 
#     early_stop_callback=early_stop, 
#     max_epochs=10,
#     auto_lr_find=True,
#     logger=comet_logger
# )
# model = CIFARTenLitModel(backbone. 1e-3)
# trainer.fit(model, train_loader, val_dataloaders=test_loader)

# Train
trainer = pl.Trainer.from_argparse_args(hparams, early_stop_callback=early_stop)
cifar_dm = CIFAR10DataModule(hparams, train_transforms=tfms, test_transforms=tfms)
model = CIFARTenLitModel(hparams)

# Find the learning rate
lr_finder = trainer.lr_find(model)
logging.info(lr_finder.results)
new_lr = lr_finder.suggestion()
model.hparams = new_lr

# Train
# trainer.fit(model, cifar_dm)