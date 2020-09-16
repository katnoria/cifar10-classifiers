from argparse import ArgumentParser

import pytorch_lightning as pl
from data import CIFAR10DataModule, tfms
from models import CIFARTenLitModel

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--workers", type=int, default=12)

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()


cifar_dm = CIFAR10DataModule(hparams, train_transforms=tfms, test_transforms=tfms)

model = pl.LightningModule.load_from_checkpoint("./cf10-pl/8751e2dd343f494c8227fcadaef62b68/checkpoints/epoch=29.ckpt")
model.freeze()

trainer = pl.Trainer.from_argparse_args(hparams)
trainer.test(model)
