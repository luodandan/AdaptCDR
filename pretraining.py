import os
import torch
import numpy as np
from itertools import chain
from models import EncoderDecoder, MLP

def eval_epoch(model, loader, device):
    model.eval()
    avg_loss = 0
    for x_batch in loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss = model.loss_function(x_batch, model(x_batch))
            avg_loss += loss.cpu().detach().item() / x_batch.size(0)
    return avg_loss

def ae_train_step(model, s_batch, t_batch, device, optimizer, scheduler=None):
    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)
    
    optimizer.zero_grad()
    model.zero_grad()
    model.train()
    
    s_loss = model.loss_function(s_x, model(s_x))
    t_loss = model.loss_function(t_x, model(t_x))
    loss = s_loss + t_loss
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.cpu().detach().item() / s_x.size(0)

def training(s_dataloaders, t_dataloaders, **kwargs):
    s_train = s_dataloaders[0]
    s_test = s_dataloaders[1]
    t_train = t_dataloaders[0]
    t_test = t_dataloaders[1]
    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         drop=kwargs['drop']).to(kwargs['device'])
    
    shared_decoder = MLP(input_dim=kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         drop=kwargs['drop']).to(kwargs['device'])
 
    thres = np.inf
    AE = EncoderDecoder(shared_encoder, shared_decoder, noise_flag=True)
    ae_optimizer = torch.optim.AdamW(AE.parameters(), lr=kwargs['lr'])
    if kwargs['retrain_flag']:
        # start dsnae pre-training
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            train_loss_all = 0
            val_loss_all = 0
            if epoch % 50 == 0:
                print(f'AE training epoch {epoch}')
            for step, s_batch in enumerate(s_train):
                t_batch = next(iter(t_train))
                train_loss = ae_train_step(model=AE, s_batch=s_batch, t_batch=t_batch,
                                                 device=kwargs['device'],
                                                 optimizer=ae_optimizer)
                train_loss_all += train_loss
                
            s_val_loss = eval_epoch(model=AE, loader=s_test, device=kwargs['device'])
            t_val_loss = eval_epoch(model=AE, loader=t_test, device=kwargs['device'])
            val_loss_all = s_val_loss + t_val_loss
            if (val_loss_all < thres):
                torch.save(AE.state_dict(), os.path.join(kwargs['model_save_folder'], 'AE.pt'))
    else:
        try:
            AE.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'AE.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")
        
    return AE.encoder
