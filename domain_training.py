# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from itertools import chain
from collections import defaultdict
from main import dict_to_str
from models import AdversarialNetwork, HDA, calc_coeff
import torch.nn as nn
import myloss 
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
                            log_loss, auc, precision_recall_curve

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, y_score=y_score)
    return auc(lr_recall, lr_precision)

def multi_eval_epoch(epoch, model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    total_loss_sum = 0.0
    y_trues, y_preds, y_masks = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_orig = y   # 原始 -1/0/1 (CPU)
            mask = (y_orig != 0).to(device)   # bool mask (batch, n_tasks)
            y_mapped = ((y_orig + 1.0) / 2.0).to(device).double()  # -1/1 -> 0/1 (0 -> 0.5 but masked)
            *_, yp = model(x)
            yp = yp.view_as(y_mapped)
            loss_per_elem = loss_fn(yp.double(), y_mapped)   # (batch, n_tasks)
            total_loss_sum += float((loss_per_elem * mask.double()).sum().item())
            y_trues.append(y_orig.cpu().numpy())
            y_preds.append(yp.cpu().numpy())
            y_masks.append(mask.cpu().numpy().astype(bool))

    avg_loss_like_original = total_loss_sum / len(loader)

    # 合并所有 batch
    y_true = np.concatenate(y_trues, axis=0)   # (N, n_tasks)
    y_pred = np.concatenate(y_preds, axis=0)   # logits (N, n_tasks)
    y_mask = np.concatenate(y_masks, axis=0)   # bool (N, n_tasks)
    
    roc_list, aupr_list, acc_list, f1_list = [], [], [], []
    for i in range(y_true.shape[1]):
        valid = y_mask[:, i]
        yi = (y_true[valid, i] + 1.0) / 2.0   # -1/1 -> 0/1
        yp_logits = y_pred[valid, i]
        yp_prob = 1.0 / (1.0 + np.exp(-yp_logits))  # sigmoid
        roc_list.append(roc_auc_score(yi, yp_prob))
        aupr_list.append(auprc(yi, yp_prob))
        y_pred_bin = (yp_prob >= 0.5).astype(int)
        acc_list.append(accuracy_score(yi.astype(int), y_pred_bin))
        f1_list.append(f1_score(yi.astype(int), y_pred_bin))

    mean_roc = float(sum(roc_list) / len(roc_list)) if roc_list else float('nan')
    all_results = [roc_list, aupr_list, acc_list, f1_list]

    return avg_loss_like_original, mean_roc, roc_list, all_results


def training(encoder, classifier, s_dataloader, t_dataloader, drug, **kwargs):
    base_network = HDA(encoder, classifier, new_cls=True, heuristic_num=4, heuristic_initial=True).to(kwargs['device'])
    ad_net = AdversarialNetwork(encoder.output_layer[0].out_features).to(kwargs['device'])
    
    parameter_list = [base_network.parameters(), ad_net.parameters()]
    optimizer = torch.optim.AdamW(chain(*parameter_list), lr=kwargs['lr'])
    #parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    #optimizer = torch.optim.AdamW(parameter_list, lr=kwargs['lr'])
    results =[]
    thres = np.inf
    best_auroc = 0
    best_auroc_list = []
    for epoch in range(500):
        train_loss_all = 0
        val_loss_all = 0
        base_network.train(True)
        ad_net.train(True)
        optimizer.zero_grad()
        for step, s_batch in enumerate(s_dataloader):
            t_batch = next(iter(t_dataloader))
            s_x = s_batch[0].to(kwargs['device'])
            s_y = s_batch[1]
            s_y = (s_y + 1.0) / 2
            s_y = s_y.to(kwargs['device'])
            t_x = t_batch[0].to(kwargs['device'])
            
            feat_s, hidden_s, focal_s, out_s = base_network(s_x, heuristic=True)
            feat_t, hidden_t, focal_t, out_t = base_network(t_x, heuristic=True)
            len_s = feat_s.size(0)
            len_t = feat_t.size(0)
        
            hiddens = torch.cat((hidden_s, hidden_t), dim=0)
            focals = torch.cat((focal_s, focal_t), dim=0)
            
            transfer_loss, heuristic = myloss.HDA_UDA(hiddens, focals, len_s, len_t, ad_net, calc_coeff(epoch))
            classifier_loss = nn.BCEWithLogitsLoss()(out_s, s_y.double())
            total_loss = transfer_loss + classifier_loss + heuristic
            total_loss.backward()
            optimizer.step()

        history = defaultdict(dict)

        val_loss, avg_auc, auroc_list, all_results = multi_eval_epoch(epoch=epoch,
                        model=base_network,
                        loader=t_dataloader,
                        device=kwargs['device'])

        if (val_loss < thres and avg_auc > best_auroc):
            best_auroc = avg_auc
            best_auroc_list = auroc_list
            torch.save(base_network.state_dict(), os.path.join(kwargs['model_save_folder'], 'HDA.pt'))    
            results = all_results
 
    return base_network, results
