# ON_SERVER = "DGX"
# ON_SERVER = "haifa"
ON_SERVER = "alsx2"

if ON_SERVER=="DGX":
    data_dir = "/workspace/repos/data/imagenette_tesselated_4000/"
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000_300imgs/"
    from src.data_stuff.pip_tools import install
    install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)
elif ON_SERVER=="haifa":
    data_dir = "/home/shatz/repos/data/tcga_data_formatted/"
    # data_dir = "/home/shatz/repos/data/imagenette_tesselated_4000/"
elif ON_SERVER=="alsx2":
    data_dir = "/tcmldrive/tcga_data_formatted/"
    # data_dir = "/home/shatz/repos/data/imagenette_tesselated_4000/"

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchvision
import argparse

from src.model_stuff.moco_model import MocoModel
from src.model_stuff.downstream_model import MyDownstreamModel
from src.data_stuff.NEW_patch_dataset import PatchDataModule
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument('--fe', type=str, default='myresnet') # lightly or myresnet
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--group_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--freeze_backbone', type=bool, default=False)
parser.add_argument('--num_epochs', type=int, default=2000)
parser.add_argument('--load_checkpoint', type=bool, default=False)
parser.add_argument('--use_dropout', type=bool, default=False)
args = parser.parse_args()

# --- hypers --- #
hypers_dict = {
        # "model_loc": "/home/shatz/repos/hrdl/saved_models/moco_model.pt",
        # /workspace/repos/hrdl/saved_models/moco/{EXP_NAME}
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=8-MOCO_train_loss_ssl=4.75.ckpt",
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=70-MOCO_train_loss_ssl=3.79.ckpt",
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=492-MOCO_train_loss_ssl=2.20.ckpt",
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=618-MOCO_train_loss_ssl=2.09.ckpt",
        "model_loc": None,
        # "fe": "lightly",
        "fe": args.fe,
        "data_dir": data_dir,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "freeze_backbone": args.freeze_backbone,
        "memory_bank_size": 4096,
        "moco_max_epochs": 250,
        "num_epochs": args.num_epochs,
        "use_dropout": args.use_dropout,
        }
# ------------- #

# make experiment name 
gs = hypers_dict["group_size"]
bs = hypers_dict["batch_size"]
eps = hypers_dict["num_epochs"]
lr = hypers_dict["learning_rate"]
freeze = hypers_dict["freeze_backbone"]
fe = hypers_dict["fe"]
lr = hypers_dict["learning_rate"]
drpout = hypers_dict["use_dropout"]
EXP_NAME = f"{ON_SERVER}_downstrexp_fe{fe}_gs{gs}_bs{bs}_lr{lr}_drpout{drpout}_freeze{freeze}"

# logger
logger=WandbLogger(project="moti_tcga_formatted", name=EXP_NAME)
logger.experiment.config.update(hypers_dict)

# monitors
lr_monitor = LearningRateMonitor(logging_interval='step')

# model
if args.load_checkpoint:
    model = MocoModel(hypers_dict["memory_bank_size"], hypers_dict["moco_max_epochs"]).load_from_checkpoint(hypers_dict["model_loc"], memory_bank_size=hypers_dict["memory_bank_size"])
else:
    model = MocoModel(hypers_dict["memory_bank_size"], hypers_dict["moco_max_epochs"])
backbone = model.feature_extractor.backbone
model = MyDownstreamModel(backbone=backbone, lr=hypers_dict["learning_rate"], num_classes=2, logger=logger, dataloader_group_size=hypers_dict["group_size"], log_everything=True, freeze_backbone=hypers_dict["freeze_backbone"], fe=hypers_dict["fe"], use_dropout=hypers_dict["use_dropout"])

# data
dm = PatchDataModule(data_dir=hypers_dict["data_dir"], batch_size=hypers_dict["batch_size"], group_size=hypers_dict["group_size"])
trainer = Trainer(gpus=1, max_epochs=hypers_dict["num_epochs"],
        logger=logger,
        # reload_dataloaders_every_epoch=True,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            PatientLevelValidation(group_size=hypers_dict["group_size"], debug_mode=False),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            ]
        )
trainer.fit(model, dm)

trainer.save_checkpoint("moco_model_newdm.pt")
