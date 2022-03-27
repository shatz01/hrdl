ON_SERVER = "DGX"
# ON_SERVER = "haifa"

if ON_SERVER=="DGX":
    data_dir = "/workspace/repos/data/imagenette_tesselated_4000/"
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000_300imgs/"
    from src.data_stuff.pip_tools import install
    install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)
elif ON_SERVER=="haifa":
    data_dir = "/home/shatz/repos/data/imagenette_tesselated/"

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchvision

from src.model_stuff.moco_model import MocoModel
from src.model_stuff.downstream_model import MyDownstreamModel
from src.data_stuff.NEW_patch_dataset import PatchDataModule
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation

# --- hypers --- #
hypers_dict = {
        # "model_loc": "/home/shatz/repos/hrdl/saved_models/moco_model.pt",
        # /workspace/repos/hrdl/saved_models/moco/{EXP_NAME}
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=8-MOCO_train_loss_ssl=4.75.ckpt",
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=70-MOCO_train_loss_ssl=3.79.ckpt",
        "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=492-MOCO_train_loss_ssl=2.20.ckpt",
        "data_dir": data_dir,
        "batch_size": 8,
        "group_size": 5,
        "memory_bank_size": 4096,
        "moco_max_epochs": 250,
        }
# ------------- #

# make experiment name 
gs = hypers_dict["group_size"]
bs = hypers_dict["batch_size"]
EXP_NAME = f"DownstreamMOCO_gs{gs}_bs{bs}_fromssl2.20"

# logger
logger=WandbLogger(project="moti_imagenette_tesselated", name=EXP_NAME)
logger.experiment.config.update(hypers_dict)

# model
model = MocoModel(hypers_dict["memory_bank_size"], hypers_dict["moco_max_epochs"]).load_from_checkpoint(hypers_dict["model_loc"], memory_bank_size=hypers_dict["memory_bank_size"])
backbone = model.feature_extractor.backbone
model = MyDownstreamModel(backbone=backbone, num_classes=2, logger=logger, dataloader_group_size=hypers_dict["group_size"], log_everything=True)

# data
dm = PatchDataModule(data_dir=hypers_dict["data_dir"], batch_size=hypers_dict["batch_size"], group_size=hypers_dict["group_size"])
trainer = Trainer(gpus=1, max_epochs=hypers_dict["moco_max_epochs"],
        logger=logger,
        callbacks=[
            PatientLevelValidation(group_size=hypers_dict["group_size"]),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            ]
        )
trainer.fit(model, dm)

trainer.save_checkpoint("moco_model_newdm.pt")
