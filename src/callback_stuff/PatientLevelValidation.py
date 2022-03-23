from collections import defaultdict
import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np

class PatientLevelValidation(pl.Callback):
    def __init__(self, group_size: int) -> None:

        print("Patient Level Eval initialized")
        # self.train_eval_dict = defaultdict(list)
        # self.val_eval_dict = defaultdict(list)
        self.all_patient_targets = {}
        self.group_size = group_size

    def setup(self, trainer, pl_module, stage=None):
        # we need the following dicts to check for label corectness
        self.train_samples_dict = trainer.datamodule.train_ds.get_samples_dict()
        self.val_samples_dict = trainer.datamodule.val_ds.get_samples_dict()

        # we need these dicts to fill with patch scores
        self.train_img_samples_score_dict = trainer.datamodule.train_ds.get_img_samples_score_dict()
        self.val_img_samples_score_dict = trainer.datamodule.val_ds.get_img_samples_score_dict()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        img_id, img_paths, y, x = batch
        batch_outputs = outputs["batch_outputs"]

        # separate patch groups
        if self.group_size > 1:
            img_paths_lol = [p.split(",") for p in img_paths]
            img_paths = [item for sublist in img_paths_lol for item in sublist]
            y = y.repeat_interleave(self.group_size)
            batch_outputs = batch_outputs.repeat_interleave(self.group_size, axis=0)
            img_id = tuple(np.repeat(np.array(img_id), self.group_size))

        self.update_dicts(img_id, img_paths, batch_outputs, y, self.train_img_samples_score_dict)


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        img_id, img_paths, y, x = batch
        batch_outputs = outputs["batch_outputs"]

        # separate patch groups
        if self.group_size > 1:
            img_paths_lol = [p.split(",") for p in img_paths]
            img_paths = [item for sublist in img_paths_lol for item in sublist]
            y = y.repeat_interleave(self.group_size)
            batch_outputs = batch_outputs.repeat_interleave(self.group_size, axis=0)
            img_id = tuple(np.repeat(np.array(img_id), self.group_size))

        self.update_dicts(img_id, img_paths, batch_outputs, y, self.val_img_samples_score_dict)


    def update_dicts(self, batch_img_ids, batch_paths, batch_scores, batch_targets, img_samples_score_dict):
        """
        Fill the patient eval dicts with patients & scores of current batch.
        curr_dict is either the trianing dict that we want to fill or the validation dict (the one to update)
        """
        # ensure lengths are correct:
        assert len(batch_paths) == len(batch_scores) and len(batch_scores) == len(batch_targets) and len(batch_img_ids) == len(batch_paths), (
        "Error. lengths are not the same")

        # fill dict
        with torch.no_grad():
            for img_id, patch_path, patch_score, patch_target in zip(batch_img_ids, batch_paths, batch_scores, batch_targets):
                img_samples_score_dict[img_id][patch_path] = patch_score


    def on_validation_epoch_end(self, trainer, pl_module):
        """ 
        Calculate Error on patient level and Clear the patient level eval dict(s),
        So that it can fill up for next epoch
        """
        # eval and record results
        if not trainer.sanity_checking:
            train_rawsum_acc, train_majority_vote_acc = self.score_dict(self.train_img_samples_score_dict, self.train_samples_dict)
            val_rawsum_acc, val_majority_vote_acc = self.score_dict(self.val_img_samples_score_dict, self.val_samples_dict)

            self.log('train_rawsum_acc', train_rawsum_acc, on_step=False, on_epoch=True)
            self.log('train_majority_vote_acc', train_majority_vote_acc, on_step=False, on_epoch=True)

            self.log('val_rawsum_acc', val_rawsum_acc, on_step=False, on_epoch=True)
            self.log('val_majority_vote_acc', val_majority_vote_acc, on_step=False, on_epoch=True)


    def score_dict(self, img_samples_score_dict, samples_dict, mode=None):
        y = []
        y_hat_rawsum = []
        y_hat_majority_vote = []
        for img_id in samples_dict.keys():
            img_y = samples_dict[img_id][1]
            patch_yhats = img_samples_score_dict[img_id]
            img_yhat = []
            for patch_path in patch_yhats:
                if patch_yhats[patch_path] is not None:
                    img_yhat.append(patch_yhats[patch_path])
            img_yhat = torch.stack(img_yhat)
            img_yhat_rawsum_logits = torch.sum(img_yhat, dim=0)
            # img_yhat_rawsum_argmax = torch.argmax(img_yhat_rawsum_logits)
            img_yhat_majority_vote = torch.mode(torch.argmax(img_yhat, dim=1)).values
            y.append(img_y)
            y_hat_rawsum.append(img_yhat_rawsum_logits)
            y_hat_majority_vote.append(img_yhat_majority_vote)
        y = torch.stack(y)
        y_hat_rawsum = torch.stack(y_hat_rawsum)
        y_hat_majority_vote = torch.stack(y_hat_majority_vote)

        rawsum_acc = torchmetrics.functional.accuracy(y_hat_rawsum.cpu(), y)
        majority_vote_acc = torchmetrics.functional.accuracy(y_hat_majority_vote.cpu(), y)

        return rawsum_acc, majority_vote_acc

