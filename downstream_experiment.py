import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchvision

from src.model_stuff.moco_model import MocoModel
from src.model_stuff.downstream_model import MyDownstreamModel
from src.data_stuff.NEW_patch_dataset import PatchDataModule
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation

# --- hypers --- #
data_dir = '/home/shatz/repos/data/imagenette_tesselated/'
batch_size = 8
group_size = 5

memory_bank_size = 4096
moco_max_epochs = 250
# ------------- #

EXP_NAME = f"Downstream_moco"
logger=WandbLogger(project="moti_imagenette_tesselated", name=EXP_NAME)
# logger = TensorBoardLogger("./lightning_logs", name=EXP_NAME)

# model
model_loc = "/home/shatz/repos/hrdl/saved_models/moco_model.pt"
model = MocoModel(memory_bank_size, moco_max_epochs).load_from_checkpoint(model_loc, memory_bank_size=memory_bank_size)
backbone = model.feature_extractor.backbone
model = MyDownstreamModel(backbone=backbone, num_classes=2, logger=logger, dataloader_group_size=group_size, log_everything=True)

# data
dm = PatchDataModule(data_dir=data_dir, batch_size=batch_size, group_size=group_size)

trainer = Trainer(gpus=1, max_epochs=moco_max_epochs,
        logger=logger,
        callbacks=[
            PatientLevelValidation(group_size=group_size),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            ]
        )

trainer.fit(model, dm)

trainer.save_checkpoint("moco_model_newdm.pt")
