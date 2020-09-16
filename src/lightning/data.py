import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models, utils

# Transforms
tfms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, hparams, data_dir = "./data", train_transforms=None, val_transforms=None, test_transforms=None):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms)
        self.data_dir = data_dir
        self.hparams = hparams

    def prepare_data(self, ):
        datasets.CIFAR10(
            root=self.data_dir, 
            train=True,
            download=True
            )
        datasets.CIFAR10(
            root=self.data_dir, 
            train=False,
            download=True
            )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar10 = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.train_transforms
            )
            self.cifar10_train, self.cifar10_val = random_split(cifar10, [45000, 5000])

        if stage == "test":
            self.cifar10_test = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.test_transforms
            )

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.cifar10_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.workers)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.cifar10_train, batch_size=32, shuffle=False, num_workers=self.hparams.workers)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.cifar10_train, batch_size=32, shuffle=False, num_workers=self.hparams.workers)