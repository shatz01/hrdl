print("-- python script started --")
ON_SERVER = "DGX"
# ON_SERVER = "haifa"
# ON_SERVER = "alsx2"

if ON_SERVER=="DGX":
    data_dir = "/workspace/repos/data/tcga_data_formatted/"
    # data_dir = "/workspace/repos/data/tcga_data_formatted_L16/" ### Patients with more than 16 patches
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000/"
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000_300imgs/"
    from src.data_stuff.pip_tools import install
    install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)
elif ON_SERVER=="haifa":
    data_dir = "/home/shatz/repos/data/tcga_data_formatted/"
    # data_dir = "/home/shatz/repos/data/imagenette_tesselated_4000/"
elif ON_SERVER=="alsx2":
    # data_dir = "/tcmldrive/tcga_data_formatted/"
    data_dir = "/tcmldrive/tcga_data_formatted_20T15V/"
    # data_dir = "/home/shatz/repos/data/imagenette_tesselated_4000/"

print(f"🚙 Starting Downstream Experiment on {ON_SERVER}! 🚗")

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

# pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument('--fe', type=str, default='lightly') # lightly or myresnet
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--group_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--freeze_backbone', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--load_checkpoint', type=bool, default=True)
parser.add_argument('--use_dropout', type=bool, default=False)
parser.add_argument('--num_FC', type=int, default=2)
parser.add_argument('--use_LRa', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_out_neurons', type=int, default=2)
args = parser.parse_args()

# --- hypers --- #
hypers_dict = {
        # "model_loc": "/home/shatz/repos/hrdl/saved_models/moco_model.pt",
        # /workspace/repos/hrdl/saved_models/moco/{EXP_NAME}
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=8-MOCO_train_loss_ssl=4.75.ckpt",
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=70-MOCO_train_loss_ssl=3.79.ckpt",
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=492-MOCO_train_loss_ssl=2.20.ckpt",
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/temp_saves/epoch=618-MOCO_train_loss_ssl=2.09.ckpt",
        "model_loc": "/workspace/repos/colorectal_cancer_ai/saved_models/epoch=510-MOCO_train_loss_ssl=0.88.ckpt", # ON DGX
        # "model_loc": "/workspace/repos/hrdl/saved_models/moco/myMOCO_150imgs_bs128_ep610_ngp1_nnodes4_stratddp_nsplits2/epoch=572-MOCO_train_loss_ssl=0.76.ckpt"
        # "model_loc": "/home/shats/repos/hrdl/saved_models/epoch=510-MOCO_train_loss_ssl=0.88.ckpt", # ON ALSX2
        # "model_loc": None,
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
        "num_FC": args.num_FC,
        "use_LRa": args.use_LRa, # learning rate annealing
        "num_workers": args.num_workers,
        "num_out_neurons": args.num_out_neurons,
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
nFC = hypers_dict["num_FC"]
LRa = hypers_dict["use_LRa"]
num_out_neurons = hypers_dict["num_out_neurons"]
EXP_NAME = f"downstrexp_{ON_SERVER}_fe{fe}_gs{gs}_bs{bs}_lr{lr}_drpout{drpout}_freeze{freeze}_nFC{nFC}_num_out_neurons{num_out_neurons}"
print(f"🚙 Experiment Name: {EXP_NAME}! 🚗")

# logger
# logger=WandbLogger(project="Equate_resnet", name=EXP_NAME)
# logger=WandbLogger(project="moti_tcga_formatted", name=EXP_NAME)
# logger=WandbLogger(project="moti_tcgaF_wROC", name=EXP_NAME)
logger=WandbLogger(project="moti_tcga_AVG100_2class", name=EXP_NAME)
logger.experiment.config.update(hypers_dict)

# monitors
lr_monitor = LearningRateMonitor(logging_interval='step')
# checkpoint_callback = ModelCheckpoint(
#     # dirpath=f'./saved_models/downstream/{EXP_NAME}',
#     dirpath=f'/workspace/repos/hrdl/saved_models/downstream/downstream10/',
#     filename='{epoch}-{val_majority_vote_acc:.3f}-{val_acc_epoch:.3f}',
#     save_top_k=1,
#     verbose=True,
#     monitor='val_majority_vote_acc',
#     mode='max'
# )

checkpoint_callback = ModelCheckpoint(
    # dirpath=f'./saved_models/downstream/{EXP_NAME}',
    dirpath=f'/workspace/repos/hrdl/saved_models/avg100ep_2class/downstream_MLP/',
    filename='{epoch}-{val_majority_vote_acc:.3f}-{val_acc_epoch:.3f}',
    verbose=True,
    monitor='epoch',
    mode='max'
)

# model
if args.load_checkpoint:
    print("♻️♻️♻️♻️♻️♻️  LOADING CHECKPOINT")
    model = MocoModel(
            hypers_dict["memory_bank_size"],
            hypers_dict["moco_max_epochs"]
            ).load_from_checkpoint(
                    hypers_dict["model_loc"], 
                    memory_bank_size=hypers_dict["memory_bank_size"])
else:
    model = MocoModel(
            hypers_dict["memory_bank_size"],
            hypers_dict["moco_max_epochs"])
backbone = model.feature_extractor.backbone
model = MyDownstreamModel(
        backbone=backbone,
        max_epochs=hypers_dict["num_epochs"],
        lr=hypers_dict["learning_rate"],
        num_classes=2,
        logger=logger,
        dataloader_group_size=hypers_dict["group_size"],
        log_everything=True,
        freeze_backbone=hypers_dict["freeze_backbone"],
        fe=hypers_dict["fe"],
        use_dropout=hypers_dict["use_dropout"],
        num_FC=hypers_dict["num_FC"],
        use_LRa=hypers_dict["use_LRa"],)
        # num_out_neurons=hypers_dict["num_out_neurons"])

# data
dm = PatchDataModule(data_dir=hypers_dict["data_dir"], batch_size=hypers_dict["batch_size"], group_size=hypers_dict["group_size"], num_workers=hypers_dict["num_workers"])
trainer = Trainer(gpus=1, max_epochs=hypers_dict["num_epochs"],
        logger=logger,
        # reload_dataloaders_every_epoch=True,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            PatientLevelValidation(group_size=hypers_dict["group_size"], debug_mode=False),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            checkpoint_callback 
            ]
        )
trainer.fit(model, dm)

# trainer.save_checkpoint("/workspace/repos/hrdl/saved_models/downstream/downstream10/{epoch}-{val_majority_vote_acc:.3f}-{val_acc_epoch:.3f}.pt")
