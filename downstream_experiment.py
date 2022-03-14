import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchvision

from src.data_stuff.patch_datamodule import TcgaDataModule
from src.data_stuff import tcga_moco_dm
from src.data_stuff.downstream_dataset import DownstreamTrainingDataset
# from src.model_stuff.MyResNet import MyResNet
from src.model_stuff.moco_model import MocoModel
from src.model_stuff.downstream_model import MyDownstreamModel
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation


import flash
from flash.image import ImageClassificationData, ImageEmbedder

# --- hypers --- #
batch_size = 16
memory_bank_size = 4096
moco_max_epochs = 250
data_dir = '/home/shatz/repos/data/imagenette_tesselated/'
min_patches_per_patient = 0
training_group_size = 5
# ------------- #

EXP_NAME = f"Downstream_moco"
logger=WandbLogger(project="moti_imagenette_tesselated", name=EXP_NAME)
# logger = TensorBoardLogger("./lightning_logs", name=EXP_NAME)

### make the embedder a patch-group level classifier
### downstream_model = 
rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
train_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(rgb_mean, rgb_std),
])
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(rgb_mean, rgb_std),
])

train_ds = DownstreamTrainingDataset(
        root_dir=data_dir, 
        transform=train_transforms, 
        dataset_type="train", 
        group_size=training_group_size, 
        subset_size=None,
        min_patches_per_patient=min_patches_per_patient
        )
val_ds = DownstreamTrainingDataset(
        root_dir=data_dir, 
        transform=val_transforms, 
        dataset_type="val", 
        group_size=training_group_size, 
        subset_size=None,
        min_patches_per_patient=min_patches_per_patient
        )
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

model_loc = "/home/shatz/repos/hrdl/saved_models/moco_model.pt"
# embedder = MocoModel(memory_bank_size, moco_max_epochs)
model = MocoModel(memory_bank_size, moco_max_epochs).load_from_checkpoint(model_loc, memory_bank_size=memory_bank_size)

backbone = model.feature_extractor.backbone
model = MyDownstreamModel(backbone=backbone, num_classes=2, logger=logger, dataloader_group_size=training_group_size, log_everything=True)


trainer = Trainer(gpus=1, max_epochs=moco_max_epochs,
        logger=logger,
        callbacks=[
            PatientLevelValidation(multi_patch=True),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            ]
        )

trainer.fit(model, train_dl, val_dl)

trainer.save_checkpoint("moco_model.pt")
