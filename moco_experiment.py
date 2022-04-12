### COMMANDS ###
# srun --pty --nodes=2 --gres=gpu:1 --container-image ~/nvidia+pytorch+21.11-py3.sqsh --container-mounts ~/repos/:/workspace/repos/ python /workspace/repos/hrdl/moco_experiment.py --num_gpus=1 --num_nodes=2 --strat=ddp







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
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data_stuff.patch_datamodule import TcgaDataModule
from src.data_stuff import tcga_moco_dm
# from src.model_stuff.MyResNet import MyResNet
from src.model_stuff.moco_model import MocoModel
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import argparse

if __name__ == "__main__":
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=700)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--strat', type=str, default=None) # dp or ddp
    args = parser.parse_args()

    # --- hypers --- #
    hypers_dict = {
            "data_dir": data_dir,
            "batch_size": args.batch_size,
            "memory_bank_size": 4096,
            "moco_max_epochs": args.num_epochs,
            "num_gpus": args.num_gpus,
            "num_nodes": args.num_nodes,
            "strat": args.strat
            }
    # min_patches_per_patient = 0
    # ------------- #

    # make experiment name
    bs = hypers_dict["batch_size"]
    ep = hypers_dict["moco_max_epochs"]
    ngp = hypers_dict["num_gpus"]
    nnodes = hypers_dict["num_nodes"]
    strat = hypers_dict["strat"]
    EXP_NAME = f"myMOCO_150imgs_bs{bs}_ep{ep}_ngp{ngp}_nnodes{nnodes}_strat{strat}"

    # logger
    print(f"------------------------------- STARTING {EXP_NAME} ----------")
    logger=WandbLogger(project="new_moti_imagenette_tesselated", name=EXP_NAME)
    # logger.experiment.config.update(hypers_dict)

    # monitors
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
            dirpath=f'/workspace/repos/hrdl/saved_models/moco/{EXP_NAME}',
        filename='{epoch}-{MOCO_train_loss_ssl:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='MOCO_train_loss_ssl',
        mode='min'
    )

    # model
    embedder = MocoModel(hypers_dict["memory_bank_size"], hypers_dict["moco_max_epochs"])
                                 
    dm = tcga_moco_dm.MocoDataModule(data_dir=hypers_dict["data_dir"],
            batch_size=hypers_dict["batch_size"], 
            subset_size=None,
            num_workers=8)

    trainer = Trainer(
            gpus=hypers_dict["num_gpus"], 
            num_nodes=hypers_dict["num_nodes"],
            strategy=hypers_dict["strat"],
            max_epochs=hypers_dict["moco_max_epochs"],
            logger=logger,
            callbacks=[
                lr_monitor,
                checkpoint_callback,
                # PatientLevelValidation.PatientLevelValidation(),
                # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
                ]
            )

    trainer.fit(embedder, dm)
