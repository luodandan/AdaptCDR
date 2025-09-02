import os
import torch
import numpy as np
from models import CorrelateEncoderDecoder, CorrelateMLP, LabelDependencySmoothing
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
                            log_loss, auc, precision_recall_curve
import config

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            trues.append(y.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    aucs = []
    for i in range(y_true.shape[1]):
        mask = (y_true[:, i] != 0)
        if mask.sum() < 2:
            continue
        yi = (y_true[mask, i] + 1) / 2.0   # -1/1 -> 0/1
        yp_i = y_pred[mask, i]
        if len(np.unique(yi)) < 2:
            continue
        aucs.append(roc_auc_score(yi, yp_i))

    return float(np.nanmean(aucs))


def corelate_classifier_train_step(model, label_smoother, batch, device, optimizer, loss_fn, scheduler=None):
    x = batch[0].to(device)
    y = batch[1]
    y = (y + 1.0) / 2.0
    y = y.to(device)

    model.zero_grad()
    optimizer.zero_grad()
    model.train()

    yp = model(x)
    smooth_loss = label_smoother(yp, y)
    loss_mat = loss_fn(yp, y.double())
    # print(loss_mat)
    loss = torch.sum(loss_mat) / x.size(0) + smooth_loss
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
    # return loss.cpu().detach().item() / x.size(0)


def multi_training(encoder, train_dataloader, val_dataloader, drug, **kwargs):
    label_graph = config.label_graph_diag
    
    classifier = CorrelateMLP(input_dim=kwargs['latent_dim'], output_dim=len(drug),
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     drop=kwargs['drop']).to(kwargs['device'])
    
    label_smoother = LabelDependencySmoothing(
            label_graph=label_graph,
            k_alpha=3,
            lambda_smooth=0.2,
            device=kwargs['device']
        ).to(kwargs['device'])
       
    predictor = CorrelateEncoderDecoder(encoder, classifier, noise_flag=False)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=kwargs['lr'])
    classifier_loss = nn.BCEWithLogitsLoss(reduction='none')
    # thres = np.inf
    best_auc = 0
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 50 == 0:
            print(f'classification training epoch {epoch}')
        for step, batch in enumerate(train_dataloader):
            corelate_classifier_train_step(model=predictor,
                                        label_smoother=label_smoother,
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