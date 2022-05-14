print("\n\n-- python script started --")
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

print(f"🚙 Starting make_ROC.py on {ON_SERVER}! 🚗\n")
import torch.multiprocessing
# above from here: https://github.com/pytorch/pytorch/issues/11201



torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchvision
import argparse

# from rich.progress import track
from tqdm import tqdm

from src.model_stuff.moco_model import MocoModel
from src.model_stuff.downstream_model import MyDownstreamModel
from src.model_stuff.downstream_model_regressor import MyDownstreamModel as MyDownstreamModelRegressor
from src.model_stuff.MyResNet import MyResNet
from src.model_stuff.MyResNetRegressor import MyResNet as MyResNetRegressor
from src.data_stuff.NEW_patch_dataset import PatchDataModule

from torchmetrics.functional.classification.roc import roc

import wandb
# from sklearn.metrics import confusion_matrix
from torchmetrics import ROC
from torchmetrics.functional import auc
import plotly
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt


def make_plot_matplotlib(record_dict):
    plt.figure()
    lw = 2
    for model_name in record_dict.keys():
        print(f"📊 Plotting {model_name} ...")
        fpr = record_dict[model_name]['fpr']
        tpr =  record_dict[model_name]['tpr']
        auc_score = record_dict[model_name]['auc']

        plt.plot(
            fpr,
            tpr,
            lw=lw,
            label=f"ROC curve (area = {auc_score:.3f})",
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig("tmp_plt.png")
    wandb.log({"img": wandb.Image('tmp_plt.png')})

def make_plot_plotly(fpr, tpr, auc_score):
    fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc_score:.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    # fig.show()
    name = "ROC"
    
    # wandb.log({name:fig})



def make_roc_image_level(preds, y_gt):
    pass

def get_roc_sample_level(preds, y_gt):
    roc = ROC(pos_label=1)
    fpr, tpr, thresholds = roc(preds, y_gt)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score


def get_preds(model, dm):
    print("... getting preds ...")
    
    dm.prepare_data()
    dm.setup()
    model.eval()
    model.cuda()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    all_preds = []
    all_y_gt = []

    for i, batch in enumerate(tqdm(val_dl)):
        with torch.no_grad():
            img_id, img_paths, y_gt, x = batch
            preds =  model.get_preds(batch)
            all_y_gt.extend(y_gt)
            all_preds.extend(preds)

    preds = torch.stack(all_preds).squeeze(-1).cpu()
    y_gt = torch.stack(all_y_gt)
    return preds, y_gt 

################################# MAIN ################################# 
def make_roc_main(models, model_names, dm):

    assert len(models) == len(model_names), "Error: You must name each model in order"
    
    record_dict = {
            # model_name = {
            #         # model: model object
            #         # preds: [0.5, 0.4, ...]
            #         # y_gt: [1, 0, ...] ... although this should be the same?
            #         # fpr: [0.0, 0.1,.. 0.9, 1.0]
            #         # tpr: [0.0, 0.1,.. 0.9, 1.0]
            #         # auc: 0.803
            #         }
            }
    # I dont think I need to store thresholds...

    for model, model_name in zip(models, model_names):
        record_dict[model_name] = {'model': model}
        print(f'♻️  Evaluating {model_name} ♻️')

        preds, y_gt = get_preds(model, dm)
        record_dict[model_name]['preds'] = preds
        record_dict[model_name]['y_gt'] = y_gt

        fpr, tpr, thresholds, auc = get_roc_sample_level(preds, y_gt)
        record_dict[model_name]['fpr'] = fpr
        record_dict[model_name]['tpr'] = tpr
        record_dict[model_name]['auc'] = auc

    make_plot_matplotlib(record_dict)















if __name__ == "__main__":
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='myresnet') # lightly or myresnet
    parser.add_argument('--checkpoint_dir', type=str,
            # lightly 
            # default='/home/shats/repos/hrdl/saved_models/downstream/alsx2_downstrexp_felightly_gs4_bs32_lr0.0001_drpoutFalse_freezeTrue_nFC2_LRaFalse/epoch=76-val_majority_vote_acc=0.87-val_acc_epoch=0.7990033626556396.ckpt',
            # default='/home/shats/repos/hrdl/saved_models/downstream_regressor/alsx2_downstrexpREGRESSOR_felightly_gs4_bs32_lr0.0001_drpoutFalse_freezeTrue_nFC2/epoch=89-val_majority_vote_acc=0.860-val_acc_epoch=0.782.ckpt',

            # resnet
            default='/home/shats/repos/hrdl/saved_models/resnet_regressor/ResnetREGRESSOR_BASELINE_alsx2_gs1_bs16/epoch=142-val_majority_vote_acc=0.89-val_acc_epoch=0.7325400114059448.ckpt'
            )

    parser.add_argument('--num_out_neurons', type=int, default=1)
    args = parser.parse_args()

    # wandb init
    EXP_NAME = f"ROC_{args.model}_outNeurons{args.num_out_neurons}"
    wandb.init(project="view_rocs", name=EXP_NAME)

    memory_bank_size = 4096
    moco_max_epochs = 0


    ####################### 💡 LIGHTLY MODEL CONFIG 💡 ####################### 
    # NOT IMPORTANT
    lr = 1e-4 # doesnt matter
    logger = None # doesnt matter
    freeze_backbone = True
    use_dropout = False
    use_LRa = False
    
    # IMPORTANT
    batch_size = 32
    dataloader_group_size = 4
    num_FC = 2
    fe = "lightly"
    
    # --- dataloader ---
    dm = PatchDataModule(
            data_dir=data_dir,
            batch_size = batch_size,
            group_size=dataloader_group_size,
            num_workers=16
            )

    # read in lightly model with checkpoint
    model = MocoModel(memory_bank_size, moco_max_epochs)
    # model = model.load_from_checkpoint(args.checkpoint_dir, memory_bank_size=memory_bank_size)
    backbone = model.feature_extractor.backbone
    if args.num_out_neurons == 2:
        print("... ♻️  loading downstream model with 2 out neurons")
        model = MyDownstreamModel(
                backbone=backbone,
                max_epochs=moco_max_epochs,
                lr=lr,
                num_classes=2,
                logger=logger,
                dataloader_group_size=dataloader_group_size,
                log_everything=True,
                freeze_backbone=freeze_backbone,
                fe=fe,
                use_dropout=use_dropout,
                num_FC=num_FC,
                use_LRa=use_LRa
                )
    elif args.num_out_neurons == 1:
        print("... ♻️  loading downstream model Regressor")
        model = MyDownstreamModelRegressor(
                backbone=backbone,
                max_epochs=moco_max_epochs,
                lr=lr,
                num_classes=2,
                logger=logger,
                dataloader_group_size=dataloader_group_size,
                log_everything=True,
                freeze_backbone=freeze_backbone,
                fe=fe,
                use_dropout=use_dropout,
                num_FC=num_FC,
                use_LRa=use_LRa
                )
    model = model.load_from_checkpoint(
            args.checkpoint_dir,
            backbone=backbone,
            max_epochs=moco_max_epochs,
            lr=lr,
            num_classes=2,
            logger=logger,
            dataloader_group_size=dataloader_group_size,
            log_everything=True,
            freeze_backbone=freeze_backbone,
            fe=fe,
            use_dropout=use_dropout,
            num_FC=num_FC,
            use_LRa=use_LRa
            )

    elif args.model == "myresnet":

        # --- dataloader ---
        dm = PatchDataModule(
                data_dir=data_dir,
                batch_size = 32,
                group_size=1,
                num_workers=16
                )

        # read in resnet model with checkpoint
        model = None
        if args.num_out_neurons == 2:
            pass
        elif args.num_out_neurons == 1:
            model = MyResNetRegressor().load_from_checkpoint(args.checkpoint_dir)

    make_roc_main(model, dm)

