import os
import torch
import numpy as np
from itertools import chain
from collections import defaultdict
from models import MLP, EncoderDecoder, IMCLNet, CorrelateEncoderDecoder, CorrelateMLP
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
                            log_loss, auc, precision_recall_curve

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)

def multi_eval_epoch(model, loader, loss_fn, device):
    model.eval()
    y_true, y_pred, y_mask = [], [], []
    roc_list = []
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            yp = model(x).squeeze(dim=1)
            # print("y===================")
            # print(y.shape)
            # print("yp==================")
            # print(yp.shape)
            # loss = loss_fn(yp, y.double())
            # avg_loss += loss.cpu().detach().item() / x.size(0)
            y_true.append(y)
            y_pred.append(yp)
            y_mask.append(mask)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    y_mask = torch.cat(y_mask, dim=0).cpu().numpy()
    # print("y_true==================================")
    # print(y_true.shape())
    # print("y_pred==================================")
    # print(y_pred)

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = (y_mask[:, i] > 0)
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
        else:
            print('{} is invalid'.format(i))

            # Y_true += y.cpu().detach().numpy().tolist()
            # Y_pred += yp.cpu().detach().numpy().tolist()
    # Y_true = np.array(Y_true)
    # Y_pred = np.array(Y_pred)
    # print(roc_auc_score(y_true=Y_true, y_score=Y_pred))
    # print(auprc(y_true=Y_true, y_score=Y_pred))
    # print(f1_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int')))
    # print(accuracy_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int')))
    
    return sum(roc_list) / len(roc_list)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    avg_loss = 0
    y_true, y_pred, y_mask = [], [], []
    roc_list = []
    for x, y, mask in loader:
        x = x.to(device)
        y = (y + 1.0)/2.0
        y = y.to(device)

        with torch.no_grad():
            yp = model(x).squeeze(dim=1)
            y_true += y.cpu().detach().numpy().tolist()
            y_pred += yp.cpu().detach().numpy().tolist()
            y_mask += mask.cpu().detach().numpy().tolist()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mask = np.array(y_mask)
    
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1.0) > 0 and np.sum(y_true[:, i] == 0.0) > 0:
            is_valid = (y_mask[:, i] > 0)
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
        else:
            print('{} is invalid'.format(i))
    # print("Y_true====================================")
    # print(Y_true)
    # print("Y_pred====================================")
    # print(Y_pred)
    # print(roc_auc_score(y_true=Y_true, y_score=Y_pred))
    # print(auprc(y_true=Y_true, y_score=Y_pred))
    # print(f1_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int')))
    # print(accuracy_score(y_true=Y_true, y_pred=(Y_pred > 0.5).astype('int')))

    return sum(roc_list)/len(roc_list)


def corelate_classifier_train_step(model, batch, device, optimizer, loss_fn, scheduler=None):
    x = batch[0].to(device)
    y = batch[1]
    y = (y + 1.0) / 2.0
    y = y.to(device)

    # print(y.shape)

    model.zero_grad()
    optimizer.zero_grad()
    model.train()

    yp = model(x)
    loss_mat = loss_fn(yp, y.double())
    # print(loss_mat)
    loss = torch.sum(loss_mat) / x.size(0)
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
    # return loss.cpu().detach().item() / x.size(0)


def multi_classifier_train_step(model, batch, device, optimizer, loss_fn, scheduler=None):
    x = batch[0].to(device)
    y = batch[1].to(device)
    mask = batch[2].to(device)
    mask = (mask > 0)
    model.zero_grad()
    optimizer.zero_grad()
    model.train()

    yp = model(x)
    loss_mat = loss_fn(yp, y.double())

    loss_mat = torch.where(
        mask, loss_mat,
        torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
    loss = torch.sum(loss_mat) / torch.sum(mask)

    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

def classifier_train_step(model, batch, device, optimizer, loss_fn, scheduler=None):
    x = batch[0].to(device)
    y = batch[1].to(device)
    model.zero_grad()
    optimizer.zero_grad()
    model.train()
    
    yp = model(x)
    loss = loss_fn(yp, y.double())
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
    return loss.cpu().detach().item() / x.size(0)


def multi_training(encoder, train_dataloader, val_dataloader, task_save_folder, drug, **kwargs):
    classifier = CorrelateMLP(input_dim=kwargs['latent_dim'], output_dim=len(drug),
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     drop=kwargs['drop']).to(kwargs['device'])
    predictor = CorrelateEncoderDecoder(encoder, classifier, noise_flag=False)

    # correlate_predictor = IMCLNet(predictor)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=kwargs['lr'])
    classifier_loss = nn.BCEWithLogitsLoss(reduction='none')

    # thres = np.inf
    best_auc = 0

    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 50 == 0:
            print(f'classification training epoch {epoch}')
        for step, batch in enumerate(train_dataloader):
            corelate_classifier_train_step(model=predictor,
                                        batch=batch,
                                        device=kwargs['device'],
                                        optimizer=optimizer,
                                        loss_fn=classifier_loss)

        avg_auc = eval_epoch(model=predictor,
                                   loader=val_dataloader,
                                   loss_fn=classifier_loss,
                                   device=kwargs['device'])


        if avg_auc > best_auc:
            torch.save(predictor.decoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'predictor.pt'))
            best_auc = avg_auc

        # if (val_loss < thres):
        #     torch.save(predictor.state_dict(), os.path.join(kwargs['model_save_folder'], 'predictor.pt'))
    return predictor.decoder

def training(encoder, train_dataloader, val_dataloader, task_save_folder, **kwargs):
    classifier = MLP(input_dim=kwargs['latent_dim'], output_dim=5,
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     drop=kwargs['drop']).to(kwargs['device'])
    predictor = EncoderDecoder(encoder, classifier, noise_flag=False)


    
    optimizer = torch.optim.AdamW(predictor.decoder.parameters(), lr=kwargs['lr'])
    classifier_loss = nn.BCEWithLogitsLoss()
    
    thres = np.inf
    if kwargs['retrain_flag']:
        for epoch in range(int(kwargs['train_num_epochs'])):
            train_loss_all = 0; val_loss_all = 0
            if epoch % 50 == 0:
                print(f'classification training epoch {epoch}') 
            for step, batch in enumerate(train_dataloader):
                train_loss = classifier_train_step(model=predictor,
                                                   batch=batch,
                                                   device=kwargs['device'],
                                                   optimizer=optimizer,
                                                   loss_fn=classifier_loss)
                train_loss_all += train_loss
                
            val_loss = eval_epoch(model=predictor,
                                  loader=val_dataloader,
                                  loss_fn=classifier_loss,
                                  device=kwargs['device'])
            if (val_loss < thres):
                torch.save(predictor.state_dict(), os.path.join(kwargs['model_save_folder'], 'predictor.pt'))
    else:
        try:
            predictor.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'predictor.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return predictor.decoder
