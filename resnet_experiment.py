ON_SERVER = "DGX"
# ON_SERVER = "haifa"

if ON_SERVER=="DGX":
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000/"
    data_dir = "/workspace/repos/data/imagenette_tesselated_4000_300imgs/"
    from src.data_stuff.pip_tools import install
    install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)
elif ON_SERVER=="haifa":
    data_dir = "/home/shatz/repos/data/imagenette_tesselated/"

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data_stuff.patch_datamodule import TcgaDataModule
from src.data_stuff.NEW_patch_dataset import PatchDataModule
from src.model_stuff.MyResNet import MyResNet
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation


# --- hypers --- #
hypers_dict = {
        "data_dir": data_dir,
        "batch_size": 64,
        "group_size": 1,
        "num_epochs": 120
        }
# ------------- #

# make experiment name
gs = hypers_dict["group_size"]
bs = hypers_dict["batch_size"]
EXP_NAME = f"Resnet_tessimagenette_gs{gs}_bs{bs}_300imgs"

# logger
logger=WandbLogger(project="moti_imagenette_tesselated", name=EXP_NAME)
logger.experiment.config.update(hypers_dict)

# model
model = MyResNet()

# data
dm = PatchDataModule(data_dir=hypers_dict["data_dir"], batch_size=hypers_dict["batch_size"], group_size=hypers_dict["group_size"])
trainer = Trainer(gpus=1, max_epochs=hypers_dict["num_epochs"],
        logger=logger,
        callbacks=[
            PatientLevelValidation(group_size=hypers_dict["group_size"]),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            ]
        )

trainer.fit(model, dm)
