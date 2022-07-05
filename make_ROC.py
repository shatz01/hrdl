import warnings
warnings.filterwarnings("ignore")

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
    data_dir = "/tcmldrive/databases/Public/tcga_data_formatted/"
    # data_dir = "/tcmldrive/shats/tcga_data_formatted_20T15V/"

print(f"🚙 Starting make_ROC.py on {ON_SERVER}! 🚗\n")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# above from here: https://github.com/pytorch/pytorch/issues/11201


import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchvision
import argparse
import numpy as np
import torchmetrics

# from rich.progress import track
from tqdm import tqdm

from src.model_stuff.moco_model import MocoModel
from src.model_stuff.downstream_model import MyDownstreamModel
from src.model_stuff.downstream_model_regressor import MyDownstreamModel as MyDownstreamModelRegressor
from src.model_stuff.downstream_model_cnn import MyDownstreamModel as MyDownstreamModelCNN
from src.model_stuff.MyResNet import MyResNet
from src.model_stuff.MyResNetRegressor import MyResNet as MyResNetRegressor
from src.data_stuff.NEW_patch_dataset import PatchDataModule

# from torchmetrics.functional.classification.roc import roc

import wandb
# from sklearn.metrics import confusion_matrix
from torchmetrics import ROC
from torchmetrics.functional import auc
import plotly
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

PLOT_TITLE = "ROC multiclass head, 1 class"
# PLOT_TITLE = "ROC multiclass head, Old method, 1 class"

# def log_accs(record_dict):
#     for model_name in record_dict.keys()
    

############################################################ MAKE PLOTS ###
def make_plot_matplotlib_sample_level(record_dict):
    plt.figure()
    lw = 2

    colors = ['C0', 'C1'] # , 'C2']
    for model_name, color in zip(record_dict.keys(), colors):
        print(f"📊 Plotting {model_name} patch level ...")
        
        # get stats
        fpr = record_dict[model_name]["interp_space"]
        tprs =  record_dict[model_name]['avg_interp_sample_tpr']
        auc_scores = record_dict[model_name]['avg_sample_auc']
        std_aucs = record_dict[model_name]['std_sample_auc']
        std_tprs = record_dict[model_name]["std_sample_tpr"]
        
        tprs_upper1 = np.minimum(tprs[0] + std_tprs[0], 1)
        tprs_upper2 = np.minimum(tprs[1] + std_tprs[1], 1)
        tprs_lower1 = np.maximum(tprs[0] - std_tprs[0], 0)
        tprs_lower2 = np.maximum(tprs[1] - std_tprs[1], 0)
        
        # # plot all lines (lightly)
        # num_models = record_dict[model_name]["num_models"]
        # for i in range(num_models):
        #     fpr_i = record_dict[model_name][f'sample_fpr{i}']
        #     tpr_i =  record_dict[model_name][f'sample_tpr{i}']
        #     plt.plot(
        #             fpr_i[0],
        #             tpr_i[0],
        #             alpha=0.5,
        #             lw=lw/2,
        #             label="ghost line"
        #             )
        #     plt.plot(
        #             fpr_i[1],
        #             tpr_i[1],
        #             alpha=0.5,
        #             lw=lw/2,
        #             ls='--',
        #             label="ghost line"
        #             )

        # plot average line
        plt.plot(
            fpr,
            tprs[0],
            lw=lw,
            color=color,
            label=f"{model_name} (auc = {auc_scores[0]:.2f} +/- {std_aucs[0]:.3f})",
        )
        # plt.plot(
        #     fpr,
        #     tprs[1],
        #     lw=lw,
        #     ls='--',
        #     label=f"{model_name} (auc = {auc_scores[0]:.2f} +/- {std_aucs[1]:.2f})",
        # )
        

        # plot std
        plt.fill_between(
            fpr,
            tprs_lower1,
            tprs_upper1,
            color=color,
            # lw=lw,
            alpha=0.3,
            label=f"{model_name} +/- 1 std dev)",
        )
        # plt.fill_between(
        #     fpr,
        #     tprs_lower2,
        #     tprs_upper2,
        #     color="grey",
        #     # lw=lw,
        #     alpha=0.3,
        #     label=f"{model_name} +/- 1 std dev)",
        # )


    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{PLOT_TITLE}, Patch Level")
    plt.legend(loc="lower right")
    plt.savefig("tmp_plt.png")
    wandb.log({"img": wandb.Image('tmp_plt.png')})

def make_plot_matplotlib_image_level(record_dict):
    plt.figure()
    lw = 2

    colors = ['C0', 'C1'] # , 'C2']
    for model_name, color in zip(record_dict.keys(), colors):
        print(f"📊 Plotting {model_name} Image level ...")
        
        # get stats
        fpr = record_dict[model_name]["interp_space"]
        tprs =  record_dict[model_name]['avg_interp_img_tpr']
        auc_scores = record_dict[model_name]['avg_img_auc']
        std_aucs = record_dict[model_name]['std_img_auc']
        std_tprs = record_dict[model_name]["std_img_tpr"]
        
        tprs_upper1 = np.minimum(tprs[0] + std_tprs[0], 1)
        tprs_upper2 = np.minimum(tprs[1] + std_tprs[1], 1)
        tprs_lower1 = np.maximum(tprs[0] - std_tprs[0], 0)
        tprs_lower2 = np.maximum(tprs[1] - std_tprs[1], 0)
        
        # plot all lines (lightly)
        num_models = record_dict[model_name]["num_models"]
        # for i in range(num_models):
        #     fpr_i = record_dict[model_name][f'img_fpr{i}']
        #     tpr_i =  record_dict[model_name][f'img_tpr{i}']
        #     plt.plot(
        #             fpr_i[0],
        #             tpr_i[0],
        #             alpha=0.5,
        #             lw=lw/2,
        #             label="ghost line"
        #             )
        #     plt.plot(
        #             fpr_i[1],
        #             tpr_i[1],
        #             alpha=0.5,
        #             lw=lw/2,
        #             ls='--',
        #             label="ghost line"
        #             )

        # plot average line
        plt.plot(
            fpr,
            tprs[0],
            lw=lw,
            color=color,
            label=f"{model_name} (auc = {auc_scores[0]:.2f} +/- {std_aucs[0]:.2f})",
        )
        # plt.plot(
        #     fpr,
        #     tprs[1],
        #     lw=lw,
        #     ls='--',
        #     label=f"{model_name} (auc = {auc_scores[0]:.2f} +/- {std_aucs[1]:.2f})",
        # )
        

        # plot std
        plt.fill_between(
            fpr,
            tprs_lower1,
            tprs_upper1,
            # lw=lw,
            color=color,
            alpha=0.3,
            label=f"{model_name} +/- 1 std dev)",
        )
        # plt.fill_between(
        #     fpr,
        #     tprs_lower2,
        #     tprs_upper2,
        #     color="grey",
        #     # lw=lw,
        #     alpha=0.3,
        #     label=f"{model_name} +/- 1 std dev)",
        # )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{PLOT_TITLE}, Patient Level")
    plt.legend(loc="lower right")
    plt.savefig("tmp_plt.png")
    wandb.log({"img": wandb.Image('tmp_plt.png')})





################################################################ GET ROC ###
def get_roc_sample_level(preds, y_gt):
    # roc = ROC(pos_label=1)
    roc = ROC(num_classes=2)
    fpr, tpr, thresholds = roc(preds, y_gt)
    auc_score = [auc(fpr, tpr) for fpr, tpr in zip(fpr, tpr)]
    return fpr, tpr, thresholds, auc_score

def get_roc_image_level(img_samples_score_dict, samples_dict_gt):

    # METHOD: sigmoid everything -> average scores per image -> take ROC
    preds = []
    y_gt = []
    for img_id in img_samples_score_dict.keys():
        try:
            all_img_scores = torch.stack([v for v in img_samples_score_dict[img_id].values() if v is not None])
        except:
            import pdb; pdb.set_trace()
        all_img_scores_sigmoided = torch.sigmoid(all_img_scores)
        all_img_scores_avg = torch.mean(all_img_scores_sigmoided, axis=0)
        preds.append(all_img_scores_avg)
        y_gt.append(samples_dict_gt[img_id][1])
        # assert samples_gt_dict[img_id] == # realized its harder to double check than i thot...

    # roc = ROC(pos_label=1)
    roc = ROC(num_classes=2)
    preds = torch.stack(preds)
    y_gt = torch.stack(y_gt)
    fpr, tpr, thresholds = roc(preds, y_gt)
    # auc_score = auc(fpr, tpr)
    auc_score = [auc(fpr, tpr) for fpr, tpr in zip(fpr, tpr)]
    return fpr, tpr, thresholds, auc_score

def fill_dict(img_samples_score_dict, preds, img_ids, img_paths):
    for img_id, patch_path, patch_score in zip(img_ids, img_paths, preds):
        img_samples_score_dict[img_id][patch_path] = patch_score

def get_preds(model, dm):
    print("... getting preds ...")
    
    model.eval()
    model.cuda()
    # train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    all_preds = []
    all_y_gt = []
    all_img_ids = []
    all_img_paths = []

    for i, batch in enumerate(tqdm(val_dl)):
        with torch.no_grad():
            img_id, img_paths, y_gt, x = batch

            preds =  model.get_preds(batch)


            if dm.group_size > 1:
                img_paths_lol = [p.split(",") for p in img_paths]
                img_paths = [item for sublist in img_paths_lol for item in sublist]
                preds = preds.repeat_interleave(dm.group_size, axis=0)
                y_gt = y_gt.repeat_interleave(dm.group_size, axis=0)
                img_id = tuple(np.repeat(np.array(img_id), dm.group_size))
                # debug: 
                # TCGA-AA-3875 is all None
                spec_id = "TCGA-AA-3875"
                # for u_img_id in set(img_id):
                #     if u_img_id not in all_img_ids:
                #         print(f"New img {u_img_id}")
                #     if u_img_id == spec_id:
                #         print("FOUND THE SPECIAL ONE")
                #         import pdb; pdb.set_trace()
            else:
                img_paths = list(img_paths[0])

            all_y_gt.extend(y_gt)
            all_preds.extend(preds)
            all_img_ids.extend(img_id)
            all_img_paths.extend(img_paths)

    preds = torch.stack(all_preds).squeeze(-1).cpu()
    y_gt = torch.stack(all_y_gt)

    return preds, y_gt, all_img_ids, all_img_paths



def get_patch_acc(preds, y_gt):
    return torchmetrics.functional.accuracy(preds, y_gt)

def get_patient_acc(img_samples_score_dict, samples_dict, mode=None):
    y = []
    y_hat_rawsum = []
    y_hat_majority_vote = []
    for img_id in samples_dict.keys():
        img_y = samples_dict[img_id][1]
        patch_yhats = img_samples_score_dict[img_id]
        img_yhat = []
        img_yhat_none_count = 0 # just for error checking
        for patch_path in patch_yhats:
            if patch_yhats[patch_path] is not None:
                img_yhat.append(patch_yhats[patch_path])
            else:
                img_yhat_none_count += 1
        try:
            img_yhat = torch.stack(img_yhat)
        except:
            import pdb; pdb.set_trace()
        # img_yhat_rawsum_argmax = torch.argmax(img_yhat_rawsum_logits)
        if img_yhat.shape[1] == 2:
            img_yhat_rawsum_logits = torch.sum(img_yhat, dim=0)
            img_yhat_majority_vote = torch.mode(torch.argmax(img_yhat, dim=1)).values
        elif img_yhat.shape[1] == 1: ### REGRESSION CASE
            img_yhat_rawsum_logits = torch.tensor(0) ### FIX THIS FOR REGRESSION CASE
            # for the above, intuition says to sigmoid them, and then just check if average is greater or less than 0.5
            img_yhat_majority_vote = torch.mode((torch.nn.functional.sigmoid(img_yhat)>0.5).type(torch.uint8).squeeze(-1)).values
        else:
            print("ERROR!!! NO fucking idea what is going on!")
        y.append(img_y)
        y_hat_rawsum.append(img_yhat_rawsum_logits)
        y_hat_majority_vote.append(img_yhat_majority_vote)
    y = torch.stack(y)
    y_hat_rawsum = torch.stack(y_hat_rawsum)
    y_hat_majority_vote = torch.stack(y_hat_majority_vote)

    rawsum_acc = torchmetrics.functional.accuracy(y_hat_rawsum.cpu(), y)
    majority_vote_acc = torchmetrics.functional.accuracy(y_hat_majority_vote.cpu(), y)

    # percent_class_MSIMUT = sum(y_hat_majority_vote==self.MSIMUT_label)/len(y_hat_majority_vote)
    # percent_class_MSIMUT = 1-sum(y_hat_majority_vote)/len(y_hat_majority_vote)

    return majority_vote_acc   # , rawsum_acc, # , percent_class_MSIMUT






################################# MAIN ################################# 
def make_roc_main(models, model_names, dms):


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
    for models, model_name, dm in zip(models, model_names, dms):

        record_dict[model_name] = {}
        # for recording patient: [scores,...]
        # train_samples_dict and val_, are for checking label correctnesss. I think I will skip
        # cause lazy
        # self.train_samples_dict = dm.train_ds.get_samples_dict()
        # val_samples_dict = dm.val_ds.get_samples_dict()
        # we need these dicts to fill with patch scores
        num_models = len(models)
        all_preds = []
        all_y_gt = []
        all_sample_fpr = []
        all_sample_tpr = []
        all_sample_auc = []
        all_img_fpr = []
        all_img_tpr = []
        all_img_auc = []
        all_patch_acc = []
        all_patient_acc = []


        for i, model in enumerate(models):
            dm.prepare_data()
            dm.setup()
            val_img_samples_score_dict = dm.val_ds.get_img_samples_score_dict() # now returns a copy
            val_samples_gt_dict = dm.val_ds.get_samples_dict()
            group_size = dm.group_size

            record_dict[model_name]['model'] = model
            record_dict[model_name]["num_models"] = num_models
            print(f'\n♻️  Evaluating {model_name} {i} ♻️')
            
            # get preds
            preds, y_gt, img_ids, img_paths = get_preds(model, dm)
            fill_dict(val_img_samples_score_dict, preds, img_ids, img_paths)
            record_dict[model_name][f'preds{i}'] = preds
            record_dict[model_name][f'y_gt{i}'] = y_gt
            record_dict[model_name][f'val_img_samples_score_dict{i}'] = val_img_samples_score_dict
            patch_acc = get_patch_acc(preds, y_gt)
            patient_acc = get_patient_acc(val_img_samples_score_dict, val_samples_gt_dict)
            record_dict[model_name][f'patch_acc{i}'] = patch_acc
            record_dict[model_name][f'patient_acc{i}'] = patient_acc
            print(f'Patch acc: {patch_acc}, Patient_acc: {patient_acc}')

            # samples level ROC stats
            sample_fpr, sample_tpr, sample_thresholds, sample_auc_score = get_roc_sample_level(preds, y_gt)
            record_dict[model_name][f'sample_fpr{i}'] = sample_fpr
            record_dict[model_name][f'sample_tpr{i}'] = sample_tpr
            record_dict[model_name][f'sample_auc{i}'] = sample_auc_score
            print(f"ROC sample level auc: {sample_auc_score}")

            # Image level ROC stats
            img_fpr, img_tpr, img_thresholds, img_auc_score = get_roc_image_level(val_img_samples_score_dict, val_samples_gt_dict)
            record_dict[model_name][f'img_fpr{i}'] = img_fpr
            record_dict[model_name][f'img_tpr{i}'] = img_tpr
            record_dict[model_name][f'img_auc{i}'] = img_auc_score
            print(f"ROC img level auc: {img_auc_score}")

            all_preds.append(preds)
            all_y_gt.append(y_gt)
            all_sample_fpr.append(sample_fpr)
            all_sample_tpr.append(sample_tpr)
            all_sample_auc.append(sample_auc_score)
            all_img_fpr.append(img_fpr)
            all_img_tpr.append(img_tpr)
            all_img_auc.append(img_auc_score)
            all_patch_acc.append(patch_acc)
            all_patient_acc.append(patient_acc)

        
        # average the numbers
        # and interpolate
        interp_space = np.linspace(0, 1, 100) # basically the new fpr (x-axis of roc)
        all_interp_sample_tpr1 = [np.interp(interp_space, fpr[0], tpr[0]) for fpr, tpr in zip(all_sample_fpr, all_sample_tpr)]
        all_interp_sample_tpr2 = [np.interp(interp_space, fpr[1], tpr[1]) for fpr, tpr in zip(all_sample_fpr, all_sample_tpr)]
        avg_interp_sample_tpr1 = np.mean(all_interp_sample_tpr1, axis=0)
        avg_interp_sample_tpr2 = np.mean(all_interp_sample_tpr2, axis=0)
        avg_sample_auc = np.mean(all_sample_auc, axis=0)
        std_sample_auc = np.std(all_sample_auc, axis=0)
        std_sample_tpr1 = np.std(all_interp_sample_tpr1, axis=0)
        std_sample_tpr2 = np.std(all_interp_sample_tpr2, axis=0)
        record_dict[model_name]["all_interp_sample_tpr"] = [all_interp_sample_tpr1, all_interp_sample_tpr2]
        record_dict[model_name]["avg_interp_sample_tpr"] = [avg_interp_sample_tpr1, avg_interp_sample_tpr2]
        record_dict[model_name]["avg_sample_auc"] = avg_sample_auc
        record_dict[model_name]["std_sample_auc"] = std_sample_auc
        record_dict[model_name]["std_sample_tpr"] = [std_sample_tpr1, std_sample_tpr2]
        
        all_interp_img_tpr1 = [np.interp(interp_space, fpr[0], tpr[0]) for fpr, tpr in zip(all_img_fpr, all_img_tpr)]
        all_interp_img_tpr2 = [np.interp(interp_space, fpr[1], tpr[1]) for fpr, tpr in zip(all_img_fpr, all_img_tpr)]
        avg_interp_img_tpr1 = np.mean(all_interp_img_tpr1, axis=0)
        avg_interp_img_tpr2 = np.mean(all_interp_img_tpr2, axis=0)
        avg_img_auc = np.mean(all_img_auc, axis=0)
        std_img_auc = np.std(all_img_auc, axis=0)
        std_img_tpr1 = np.std(all_interp_img_tpr1, axis=0)
        std_img_tpr2 = np.std(all_interp_img_tpr2, axis=0)
        record_dict[model_name]["all_interp_img_tpr"] = [all_interp_img_tpr1, all_interp_img_tpr2]
        record_dict[model_name]["avg_interp_img_tpr"] = [avg_interp_img_tpr1, avg_interp_img_tpr2]
        record_dict[model_name]["avg_img_auc"] = avg_img_auc
        record_dict[model_name]["std_img_auc"] = std_img_auc
        record_dict[model_name]["std_img_tpr"] = [std_img_tpr1, std_img_tpr2]

        avg_patch_acc = np.mean(all_patch_acc)
        patch_std = np.std(all_patch_acc)
        avg_patient_acc = np.mean(all_patient_acc)
        patient_std = np.std(all_patient_acc)
        print(f'🧮 AVG patch acc: {avg_patch_acc} --- AVG patient acc: {avg_patient_acc} ---')
        print(f'🧮 patch STD: {patch_std} --- patient STD: {patient_std} ---')
        record_dict[model_name]["avg_patch_acc"] = avg_patch_acc
        record_dict[model_name]["avg_patient_acc"] = avg_patient_acc

        record_dict[model_name]["interp_space"] = interp_space



    make_plot_matplotlib_sample_level(record_dict)
    make_plot_matplotlib_image_level(record_dict)
    log_accs(record_dict)















if __name__ == "__main__":
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_out_neurons', type=int, default=2)
    # parser.add_argument('--inference_level', type=str, default="patch") # "patch" or "image"
    args = parser.parse_args()

    # wandb init
    EXP_NAME = "ROC"
    # wandb.init(project="view_rocs", name=EXP_NAME)
    # wandb.init(project="moti_tcga_AVG10", name=EXP_NAME)
    # wandb.init(project="moti_tcga_AVG100_2class", name=EXP_NAME)

    
    # checkpoint dirs
    lightly_checkpoint_1 = '/home/shats/repos/hrdl/saved_models/downstream_MLP_L2_wd1e-06/epoch=99-val_majority_vote_acc=0.860-val_acc_epoch=0.789.ckpt'
    lightly_checkpoint_2 = '/home/shats/repos/hrdl/saved_models/downstream_MLP_L2_wd1e-06/epoch=99-val_majority_vote_acc=0.860-val_acc_epoch=0.792-v1.ckpt'
    lightly_checkpoint_3 = '/home/shats/repos/hrdl/saved_models/downstream_MLP_L2_wd1e-06/epoch=99-val_majority_vote_acc=0.860-val_acc_epoch=0.792.ckpt'
    lightly_checkpoint_4 = '/home/shats/repos/hrdl/saved_models/downstream_MLP_L2_wd1e-06/epoch=99-val_majority_vote_acc=0.860-val_acc_epoch=0.800.ckpt'
    lightly_checkpoint_5 = '/home/shats/repos/hrdl/saved_models/downstream_MLP_L2_wd1e-06/epoch=99-val_majority_vote_acc=0.860-val_acc_epoch=0.802.ckpt'
    # lightly_checkpoint_6 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/downstream_MLP_L2_wd1e06/'
    lightly_checkpoints = [lightly_checkpoint_1, lightly_checkpoint_2, lightly_checkpoint_3, lightly_checkpoint_4, lightly_checkpoint_5]

    resnet_checkpoint_1 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/resnet/epoch=99-val_majority_vote_acc=0.810-val_acc_epoch=0.715.ckpt'
    resnet_checkpoint_2 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/resnet/epoch=99-val_majority_vote_acc=0.840-val_acc_epoch=0.705.ckpt'
    resnet_checkpoint_3 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/resnet/epoch=99-val_majority_vote_acc=0.840-val_acc_epoch=0.720.ckpt'
    resnet_checkpoint_4 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/resnet/epoch=99-val_majority_vote_acc=0.840-val_acc_epoch=0.731.ckpt'
    resnet_checkpoint_5 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/resnet/epoch=99-val_majority_vote_acc=0.840-val_acc_epoch=0.737.ckpt'
    resnet_checkpoint_6 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/resnet/epoch=99-val_majority_vote_acc=0.850-val_acc_epoch=0.719.ckpt'
    resnet_checkpoints = [resnet_checkpoint_1, resnet_checkpoint_2, resnet_checkpoint_3, resnet_checkpoint_4, resnet_checkpoint_5, resnet_checkpoint_6]

    # cnn_checkpoint_1 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/downstream_CNN/epoch=99-val_majority_vote_acc=0.810-val_acc_epoch=0.743.ckpt'
    # cnn_checkpoint_2 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/downstream_CNN/epoch=99-val_majority_vote_acc=0.810-val_acc_epoch=0.765.ckpt'
    # cnn_checkpoint_3 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/downstream_CNN/epoch=99-val_majority_vote_acc=0.850-val_acc_epoch=0.746.ckpt'
    # cnn_checkpoint_4 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/downstream_CNN/epoch=99-val_majority_vote_acc=0.850-val_acc_epoch=0.752.ckpt'
    # cnn_checkpoint_5 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/downstream_CNN/epoch=99-val_majority_vote_acc=0.850-val_acc_epoch=0.784.ckpt'
    # cnn_checkpoint_6 = '/home/shats/repos/hrdl/saved_models/avg100ep_2class/downstream_CNN/epoch=99-val_majority_vote_acc=0.860-val_acc_epoch=0.755.ckpt'
    # cnn_checkpoints = [cnn_checkpoint_1, cnn_checkpoint_2, cnn_checkpoint_3, cnn_checkpoint_4, cnn_checkpoint_5, cnn_checkpoint_6]

    ####################### 💡 LIGHTLY MODEL CONFIG 💡 ####################### 
    # NOT IMPORTANT
    memory_bank_size = 4096
    moco_max_epochs = 0
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
    downstream_dm = PatchDataModule(
            data_dir=data_dir,
            batch_size = batch_size,
            group_size=dataloader_group_size,
            num_workers=16
            )

    # read in lightly model with checkpoint
    downstream_model = MocoModel(memory_bank_size, moco_max_epochs)
    # model = model.load_from_checkpoint(args.checkpoint_dir, memory_bank_size=memory_bank_size)
    backbone = downstream_model.feature_extractor.backbone
    if args.num_out_neurons == 2:
        print("♻️  Loading downstream model with 2 out neurons ...", end='')
        downstream_model = MyDownstreamModel(
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
        print("♻️  Loading downstream model Regressor ...", end='')
        downstream_model = MyDownstreamModelRegressor(
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
    downstream_models = [downstream_model.load_from_checkpoint(lightly_checkpoint,backbone=backbone,max_epochs=moco_max_epochs,lr=lr,num_classes=2,logger=logger,dataloader_group_size=dataloader_group_size,log_everything=True,freeze_backbone=freeze_backbone,fe=fe,use_dropout=use_dropout,num_FC=num_FC,use_LRa=use_LRa) for lightly_checkpoint in lightly_checkpoints]
    print(' ... Done :) ')


    ####################### 💡 RESNET MODEL CONFIG 💡 ####################### 
    # --- dataloader ---
    print("♻️  Loading Resnet...", end='')
    resnet_dm = PatchDataModule(
            data_dir=data_dir,
            batch_size = 32,
            group_size=1,
            num_workers=16
            )
    # read in resnet model with checkpoint
    resnet_model = None
    if args.num_out_neurons == 2:
        resnet_models = [MyResNet().load_from_checkpoint(resnet_checkpoint) for resnet_checkpoint in resnet_checkpoints]
    elif args.num_out_neurons == 1:
        resnet_models = [MyResNetRegressor().load_from_checkpoint(resnet_checkpoint) for resnet_checkpoint in resnet_checkpoints]
        # resnet_model = MyResNetRegressor().load_from_checkpoint(resnet_checkpoint)
    print("... Done :)")
    print("✅ Done Loading Models")

    ####################### 💡 CNN MODEL CONFIG 💡 ####################### 
    # NOT IMPORTANT
    memory_bank_size = 4096
    moco_max_epochs = 0
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
    downstream_cnn_dm = PatchDataModule(
            data_dir=data_dir,
            batch_size = batch_size,
            group_size=dataloader_group_size,
            num_workers=16
            )

    # read in lightly model with checkpoint
    downstream_cnn_model = MocoModel(memory_bank_size, moco_max_epochs)
    # model = model.load_from_checkpoint(args.checkpoint_dir, memory_bank_size=memory_bank_size)
    backbone = downstream_cnn_model.feature_extractor.backbone
    if args.num_out_neurons == 2:
        print("♻️  Loading downstream model CNN with 2 out neurons ...", end='')
        downstream_cnn_model = MyDownstreamModelCNN(
                backbone=backbone,
                max_epochs=moco_max_epochs,
                lr=lr,
                num_classes=2,
                logger=logger,
                dataloader_group_size=dataloader_group_size,
                log_everything=True,
                freeze_backbone=freeze_backbone,
                )
    elif args.num_out_neurons == 1:
        print("♻️  Loading downstream model CNN Regressor ...", end='')
        raise NotImplementedError
        # downstream_model = MyDownstreamModelRegressor(
        #         backbone=backbone,
        #         max_epochs=moco_max_epochs,
        #         lr=lr,
        #         num_classes=2,
        #         logger=logger,
        #         dataloader_group_size=dataloader_group_size,
        #         log_everything=True,
        #         freeze_backbone=freeze_backbone,
        #         fe=fe,
        #         use_dropout=use_dropout,
        #         num_FC=num_FC,
        #         use_LRa=use_LRa
        #         )
    # downstream_cnn_models = [downstream_cnn_model.load_from_checkpoint(cnn_checkpoint,backbone=backbone,max_epochs=moco_max_epochs,lr=lr,num_classes=2,logger=logger,dataloader_group_size=dataloader_group_size,log_everything=True,freeze_backbone=freeze_backbone,fe=fe,use_dropout=use_dropout,num_FC=num_FC,use_LRa=use_LRa) for cnn_checkpoint in cnn_checkpoints]
    print(' ... Done :) ')







    ####################### 💡 RUN ROC MAIN 💡 ####################### 
    # models = [downstream_models]
    # model_names = ["Downstream_MOCO"]
    # dms = [downstream_dm]

    # models = [resnet_models]
    # model_names = ["Resnet"]
    # dms = [resnet_dm]

    # models = [downstream_cnn_models]
    # model_names = ["CNN_head"]
    # dms = [downstream_cnn_dm]

    models = [resnet_models, downstream_models]
    model_names = ["Resnet", "Downstream_MOCO"]
    dms = [resnet_dm, downstream_dm]

    # models = [resnet_models, downstream_models, downstream_cnn_models]
    # model_names = ["Resnet", "Downstream_MOCO", "CNN_head"]
    # dms = [resnet_dm, downstream_dm, downstream_cnn_dm]
    
    make_roc_main(models, model_names, dms)

