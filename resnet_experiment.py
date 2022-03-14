import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data_stuff.patch_datamodule import TcgaDataModule
from src.model_stuff.MyResNet import MyResNet
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation

# --- hypers --- #
data_dir = '/home/shatz/repos/data/imagenette_tesselated/'
min_patches_per_patient = 0
# ------------- #

EXP_NAME = f"imagnette_tesselated_resnet"
logger=WandbLogger(project="moti_imagenette_tesselated", name=EXP_NAME)
# logger = TensorBoardLogger("./lightning_logs", name=EXP_NAME)

model = MyResNet()
dm = TcgaDataModule(data_dir=data_dir, batch_size=64, fast_subset=False, min_patches_per_patient=min_patches_per_patient)
# class_to_idx = dm.get_class_to_idx_dict()

trainer = Trainer(gpus=1, max_epochs=120,
        logger=logger,
        callbacks=[
            PatientLevelValidation(),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            ]
        )

trainer.fit(model, dm)
