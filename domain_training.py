# -*- coding: utf-8 -*-
import json
import os

import pandas as pd
import torch
import numpy as np
from itertools import chain
from collections import defaultdict

from main import dict_to_str
from models import MLP, EncoderDecoder, AdversarialNetwork, HDA, calc_coeff
import torch.nn as nn
import Myloss 
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
                            log_loss, auc, precision_recall_curve

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)


def eval_epoch_save(model, loader, device, task_save_folder, param_list):
    model.eval()
    avg_loss = 0
    Y_true, Y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, _, _, yp = model(x)
            loss = nn.BCEWithLogitsLoss()(yp, y.double())
            avg_loss += loss.cpu().detach().item() / x.size(0)
            Y_true += y.cpu().detach().numpy().tolist()
            Y_pred += yp.cpu().detach().numpy().tolist()
    Y_true = np.array(Y_true)
    Y_pred = np.array(Y_pred)

    auroc = roc_auc_score(y_true=Y_true, y_score=Y_pred)
    aurpc = auprc(y_true=Y_true, y_score=Y_pred)
    f1 = f1_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int'))
    acc = accuracy_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int'))
    metrics = dict()
    metrics['auroc'] = auroc
    metrics['aurpc'] = aurpc
    metrics['f1'] = f1
    metrics['acc'] = acc
    print(metrics)
    param_str = dict_to_str(param_list)
    with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(metrics, f)

    return avg_loss

def multi_eval_epoch(epoch, model, loader, drug, history, device):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_mask = [], [], []
    roc_list, aupr_list, acc_list, f1_list = [], [], [], []
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        y = (y + 1.0) / 2.0
        mask = mask.to(device)
        mask = (mask > 0)
        with torch.no_grad():
            _, _, _, yp = model(x)
            # print(yp)
            loss_mat = nn.BCEWithLogitsLoss()(yp, y.double())
            loss_mat = torch.where(
                mask, loss_mat,
                torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(mask)
            total_loss += loss
            y_true.append(y)
            y_pred.append(yp)
            y_mask.append(mask)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    y_mask = torch.cat(y_mask, dim=0).cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1.0) > 0 and np.sum(y_true[:, i] == 0.0) > 0:
            is_valid = (y_mask[:, i] > 0)
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
            aupr_list.append(auprc(y_true[is_valid, i], y_pred[is_valid, i]))
            f1_list.append(f1_score(y_true[is_valid, i], (y_pred[is_valid, i]>=0.5).astype('int')))
            acc_list.append(accuracy_score(y_true[is_valid, i], (y_pred[is_valid, i]>=0.5).astype('int')))
        else:
            print('{} is invalid'.format(i))
            
    #print('epoch: {}\tavg loss: {:.6f}\tauc: {:.6f}\tauc_list: {}'.format(epoch, (total_loss/len(loader)), sum(roc_list)/len(roc_list), str(roc_list)))
    all_results=[roc_list, aupr_list, acc_list, f1_list]
    return total_loss/len(loader), sum(roc_list)/len(roc_list), roc_list, y_true, y_pred, y_mask, all_results
            
    # Y_true = np.array(Y_true)
    # Y_pred = np.array(Y_pred)
    # print(roc_auc_score(y_true=Y_true, y_score=Y_pred))
    # print(auprc(y_true=Y_true, y_score=Y_pred))
    # print(f1_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int')))
    # print(accuracy_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int')))

    # return sum(roc_list) / len(roc_list)


def eval_epoch(epoch, model, loader, drug, history, device):
    model.eval()
    avg_loss = 0
    Y_true, Y_pred, Y_mask = [], [], []
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            _, _, _, yp = model(x)
            loss = nn.BCEWithLogitsLoss()(yp, y.double())
            avg_loss += loss.cpu().detach().item() / x.size(0)
            Y_true += y.cpu().detach().numpy().tolist()
            Y_pred += yp.cpu().detach().numpy().tolist()
            Y_mask += mask.cpu().detach().numpy().tolist()


    for i in range(len(drug)):
        preds = []
        truths = []
        for j in range(len(Y_mask)):
            if Y_mask[j][i] != 0.0:
                preds.append(Y_pred[j][i])
                truths.append(Y_true[j][i])

        assert len(preds) == len(truths)

        preds = np.array(preds)
        truths = np.array(truths)

        if drug[i] not in history.keys():
            history[drug[i]] = defaultdict()
        history[drug[i]]['acc'] = accuracy_score(y_true=truths, y_pred=(preds > 0.5).astype('int'))
        history[drug[i]]['auroc'] = roc_auc_score(y_true=truths, y_score=preds)
        history[drug[i]]['aps'] = average_precision_score(y_true=truths, y_score=preds)
        history[drug[i]]['f1'] = f1_score(y_true=truths, y_pred=(preds > 0.5).astype('int'))
        history[drug[i]]['bce'] = log_loss(y_true=truths, y_pred=preds)
        history[drug[i]]['auprc'] = auprc(y_true=truths, y_score=preds)

    print(history)

    # auroc = roc_auc_score(y_true=Y_true, y_score=Y_pred)
    # aurpc = auprc(y_true=Y_true, y_score=Y_pred)
    # f1 = f1_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int'))
    # acc = accuracy_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int'))
    # metrics = dict()
    # metrics['avg_loss'] = avg_loss
    # metrics['auroc'] = auroc
    # metrics['aurpc'] = aurpc
    # metrics['f1'] = f1
    # metrics['acc'] = acc
    # print(metrics)
    
    return avg_loss, history

def training(encoder, classifier, s_dataloader, t_dataloader, drug, task_save_folder, params_str, **kwargs):
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
    for epoch in range(300):
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
            feats = torch.cat((feat_s, feat_t), dim=0)
            hiddens = torch.cat((hidden_s, hidden_t), dim=0)
            focals = torch.cat((focal_s, focal_t), dim=0)
            outs = torch.cat((out_s, out_t), dim=0)
            
            transfer_loss, heuristic = Myloss.HDA_UDA(hiddens, focals, len_s, len_t, ad_net, calc_coeff(epoch))
            classifier_loss = nn.BCEWithLogitsLoss()(out_s, s_y.double())
            total_loss = transfer_loss + classifier_loss + heuristic #+ config["gauss"] *gaussian
            total_loss.backward()
            optimizer.step()

        history = defaultdict(dict)

        val_loss, avg_auc, auroc_list, y_true, y_pred, y_mask, all_results = multi_eval_epoch(epoch=epoch,
                        model=base_network,
                        loader=t_dataloader,
                        drug=drug,
                        history=history,
                        device=kwargs['device'])

        if (val_loss < thres and avg_auc > best_auroc):
            best_auroc = avg_auc
            best_auroc_list = auroc_list
            torch.save(base_network.state_dict(), os.path.join(kwargs['model_save_folder'], 'HDA.pt'))
            np.save(os.path.join(kwargs['model_save_folder'], 'y_true_1111.npy'), y_true)
            np.save(os.path.join(kwargs['model_save_folder'], 'y_pred_1111.npy'), y_pred)
            np.save(os.path.join(kwargs['model_save_folder'], 'y_mask.npy'), y_mask)          
            results = all_results

    with open("{}_auroc.txt".format(params_str), 'a') as opf:
        opf.write("\t".join(list(map(str, best_auroc_list))))
        opf.write('\n')
    opf.close()
 
    return base_network, results
