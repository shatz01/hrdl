print("-- python script started --")
# ON_SERVER = "DGX"
# ON_SERVER = "haifa"
ON_SERVER = "alsx2"

if ON_SERVER=="DGX":
    data_dir = "/workspace/repos/data/tcga_data_formatted/"
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000/"
    # data_dir = "/workspace/repos/data/imagenette_tesselated_4000_300imgs/"
    from src.data_stuff.pip_tools import install
    install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)
elif ON_SERVER=="haifa":
    data_dir = "/home/shatz/repos/data/tcga_data_formatted/"
    # data_dir = "/home/shatz/repos/data/imagenette_tesselated_4000/"
elif ON_SERVER=="alsx2":
    data_dir = "/tcmldrive/tcga_data_formatted/"
    # data_dir = "/home/shatz/repos/data/imagenette_tesselated_4000/"

print(f"🚙 Starting make_ROC on {ON_SERVER}! 🚗")

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchvision
import argparse

def make_roc(model):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightly') # lightly or myresnet
    args = parser.parse_args()

    if args.model == "lightly":
        # read in lightly model with checkpoint
        model = MocoModel(hypers_dict["memory_bank_size"], hypers_dict["moco_max_epochs"])
        model = model.load_from_checkpoint(hypers_dict["model_loc"], memory_bank_size=hypers_dict["memory_bank_size"])
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
                use_LRa=hypers_dict["use_LRa"]
                )

    elif args.model == "myresnet":
        # read in resnet model with checkpoint
        model = None

    make_roc(model)

