print("-- python script started --")
ON_SERVER = "DGX"
# ON_SERVER = "haifa"
# ON_SERVER = "alsx2"

if ON_SERVER=="DGX":
    data_dir = "/workspace/repos/data/tcga_data_formatted/"
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000/"
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000_300imgs/"
    from src.data_stuff.pip_tools import install
    install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)
elif ON_SERVER=="haifa":
    data_dir = "/home/shatz/repos/data/imagenette_tesselated_4000/"
elif ON_SERVER=="alsx2":
    data_dir = "/tcmldrive/tcga_data_formatted/"
    # data_dir = "/home/shats/data/tcga_data_formatted/"
    # data_dir = "/home/shatz/repos/data/imagenette_tesselated_4000/"

print(f"🚙 Starting Resnet Experiment Regressor on {ON_SERVER}! 🚗")

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import argparse

# from src.data_stuff.patch_datamodule import TcgaDataModule
from src.data_stuff.NEW_patch_dataset import PatchDataModule
from src.model_stuff.MyResNetRegressor import MyResNet
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation

# pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--group_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

# --- hypers --- #
hypers_dict = {
        "data_dir": data_dir,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers
        }
# ------------- #

# make experiment name
gs = hypers_dict["group_size"]
bs = hypers_dict["batch_size"]
EXP_NAME = f"ResnetREGRESSOR_BASELINE_{ON_SERVER}_gs{gs}_bs{bs}"
print(f"🚙 Experiment Name: {EXP_NAME}! 🚗")

# logger
# logger=WandbLogger(project="Equate_resnet", name=EXP_NAME)
# logger=WandbLogger(project="moti_tcga_formatted", name=EXP_NAME)
# logger=WandbLogger(project="moti_tcgaF_wROC", name=EXP_NAME)
logger=WandbLogger(project="moti_tcga_AVG100_2class", name=EXP_NAME)
logger.experiment.config.update(hypers_dict)

# monitor
# checkpoint_callback = ModelCheckpoint(
#     dirpath=f'./saved_models/resnet_regressor/{EXP_NAME}',
#     filename='{epoch}-{val_majority_vote_acc:.2f}-{val_acc_epoch}',
#     save_top_k=3,
#     verbose=True,
#     monitor='val_majority_vote_acc',
#     mode='max'
# )
checkpoint_callback = ModelCheckpoint(
    # dirpath=f'./saved_models/downstream/{EXP_NAME}',
    dirpath=f'/workspace/repos/hrdl/saved_models/avg100ep_1class/resnet/',
    filename='{epoch}-{val_majority_vote_acc:.3f}-{val_acc_epoch:.3f}',
    verbose=True,
    monitor='epoch',
    mode='max'
)

# model
model = MyResNet()

# data
dm = PatchDataModule(data_dir=hypers_dict["data_dir"], batch_size=hypers_dict["batch_size"], group_size=hypers_dict["group_size"], num_workers=hypers_dict["num_workers"])

# trainer
trainer = Trainer(gpus=1, max_epochs=hypers_dict["num_epochs"],
        logger=logger,
        callbacks=[
            PatientLevelValidation(group_size=hypers_dict["group_size"], debug_mode=False),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            checkpoint_callback
            ]
        )

trainer.fit(model, dm)
