from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CometLogger

pl.seed_everything(42)

comet_logger = CometLogger(
    api_key='r3QI6mx4KaB3v0VMFwt6bcf33',
    workspace='katnoria',  # Optional
    save_dir='.',  # Optional
    project_name='cf10-pl',  # Optional
    experiment_name='pre-2'  # Optional
)

tfms = transforms.Compose([
    transforms.Resize(224),    
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

train_ds = datasets.CIFAR10(
    root="./data", 
    train=True,
    download=True,
    transform=tfms
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=12)

test_ds = datasets.CIFAR10(
    root="./data", 
    train=False,
    download=True,
    transform=tfms
)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=12)

backbone = models.resnet50(pretrained=True)
for param in backbone.parameters():
    param.requires_grad = False

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)

trainer = pl.Trainer(
    fast_dev_run=False, 
    gpus=1, 
    early_stop_callback=early_stop, 
    max_epochs=10,
    auto_lr_find=True,
    logger=comet_logger
)

model = CIFARTenLitModel(backbone. 1e-3)
trainer.fit(model, train_loader, val_dataloaders=test_loader)