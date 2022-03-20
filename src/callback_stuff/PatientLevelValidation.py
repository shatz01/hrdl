from collections import defaultdict
import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np

class PatientLevelValidation(pl.Callback):
    def __init__(self, group_size: int) -> None:

        print("Patient Level Eval initialized")
        self.train_eval_dict = defaultdict(list)
        self.val_eval_dict = defaultdict(list)
        self.all_patient_targets = {}
        self.group_size = group_size

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        img_id, img_paths, y, x = batch
        batch_outputs = outputs["batch_outputs"]
        self.patient_eval(paths, batch_outputs, y, 'train')


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

        self.patient_eval(img_id, img_paths, batch_outputs, y, self.val_eval_dict )


    def patient_eval(self, batch_img_ids, batch_paths, batch_scores, batch_targets, curr_dict):
        """
        Fill the patient eval dicts with patients & scores of current batch.
        curr_dict is either tra
        """
        # ensure lengths are correct:
        assert len(batch_paths) == len(batch_scores) and len(batch_scores) == len(batch_targets) and len(batch_img_ids) == len(batch_paths), (
        "Error. lengths are not the same" )
        import pdb; pdb.set_trace()


        # fill dict
        # with torch.no_grad():
        #     for for i in zip(batch_img_ids, batch_paths, batch_scores, batch_targets):


    def on_validation_epoch_end(self, trainer, pl_module):
        """ 
        Calculate Error on patient level and Clear the patient level eval dict(s),
        So that it can fill up for next epoch
        """        
        # eval and record results
        # need to loop over dicts to make this to ensure order is correct
        # dicts dont necessarily enforce order
        if len(self.train_patient_eval_dict) > 0:
            train_patient_scores = []
            train_patient_targets = []
            for patient in self.train_patient_eval_dict.keys():
                train_patient_score = sum(self.train_patient_eval_dict[patient])/len(self.train_patient_eval_dict[patient])
                train_patient_score = train_patient_score.clone().detach()
                train_patient_target = self.all_patient_targets[patient].clone().detach()
                train_patient_scores.append(train_patient_score)
                train_patient_targets.append(train_patient_target)
            
            train_patient_scores = torch.stack(train_patient_scores)
            train_patient_targets = torch.stack(train_patient_targets)
            train_loss = pl_module.criteria(train_patient_scores, torch.nn.functional.one_hot(train_patient_targets, num_classes=2).float())
            train_acc = torchmetrics.functional.accuracy(torch.argmax(train_patient_scores, dim=1), train_patient_targets)
            self.log('train_patientlvl_loss', train_loss)
            self.log('train_patientlvl_acc', train_acc)

        if len(self.val_patient_eval_dict) > 0:
            val_patient_scores = []
            val_patient_targets = []
            for patient in self.val_patient_eval_dict.keys():
                val_patient_score = sum(self.val_patient_eval_dict[patient])/len(self.val_patient_eval_dict[patient])
                val_patient_score = val_patient_score.clone().detach()
                val_patient_target = self.all_patient_targets[patient].clone().detach()
                val_patient_scores.append(val_patient_score)
                val_patient_targets.append(val_patient_target)

            val_patient_scores = torch.stack(val_patient_scores)
            val_patient_targets = torch.stack(val_patient_targets)
            val_loss = pl_module.criteria(val_patient_scores, torch.nn.functional.one_hot(val_patient_targets, num_classes=2).float())
            val_acc = torchmetrics.functional.accuracy(torch.argmax(val_patient_scores, dim=1), val_patient_targets)
            self.log('val_patientlvl_loss', val_loss)
            self.log('val_patientlvl_acc', val_acc)

        self.train_patient_eval_dict = defaultdict(list)
        self.val_patient_eval_dict = defaultdict(list)
 
