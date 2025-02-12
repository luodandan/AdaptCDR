import numpy as np
import pandas as pd
import torch
import json
import os
import argparse
import pickle
from collections import defaultdict
import itertools
import data
import data_config
import pretraining, training_classifier, domain_training
from copy import deepcopy
import time

def wrap_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])
    return aux_dict

def make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])

def main(args, drug, params_dict):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #load mix gene expressions for both 9000 tcga and 1000 cell line
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)
    #load traning params
    with open(os.path.join('train_params.json'), 'r') as f:
        training_params = json.load(f)
    training_params['unlabeled'].update(params_dict)
    param_str = dict_to_str(params_dict)
    method_save_folder = os.path.join('model_save')

    training_params.update({'device': device, 
                            'input_dim': gex_features_df.shape[-1], 
                            'model_save_folder': os.path.join(method_save_folder, param_str),
                            'retrain_flag': args.retrain_flag, 
                            'norm_flag': args.norm_flag})

    task_save_folder = os.path.join(f'{method_save_folder}', args.metric)
    make_dir(training_params['model_save_folder'])
    make_dir(task_save_folder)

    s_dataloaders, t_dataloaders = data.get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2023,
        batch_size=training_params['unlabeled']['batch_size'],
        ccle_only=False)
    
    labeled_dataloader = data.get_multi_labeled_dataloader_generator(gex_features_df=gex_features_df,
        seed=2023,
        batch_size=training_params['labeled']['batch_size'],
        drug=drug,
        ccle_measurement=args.measurement,
        threshold_list=args.a_thres,
        days_threshold_list=args.days_thres,
        n_splits=args.n)

    with open("{}_auroc.txt".format(param_str), 'a') as opf:
        opf.write(param_str)
        opf.write('\n')
    opf.close()

    # start unlabeled training, obtain target shared encoder
    encoder = pretraining.training(s_dataloaders=s_dataloaders,
                                   t_dataloaders=t_dataloaders,
                                   **wrap_params(training_params,
                                                 type='unlabeled'))
   
    fold = 0
    for train_labeled_ccle, test_labeled_ccle, labeled_tcga in labeled_dataloader:            
        ft_encoder = deepcopy(encoder)
        print('--------------------', drug)
        print('Cline training samples:', train_labeled_ccle.dataset.tensors[1].shape)
        print('Cline testing samples:', test_labeled_ccle.dataset.tensors[1].shape)
        print('TCGA testing samples:', labeled_tcga.dataset.tensors[1].shape)
        classifier = training_classifier.multi_training(
                                            encoder=ft_encoder,
                                            train_dataloader=train_labeled_ccle,
                                            val_dataloader=test_labeled_ccle,
                                            task_save_folder=task_save_folder,
                                            drug=drug,
                                            **wrap_params(training_params, type='labeled'))

        classifier.load_state_dict(torch.load(os.path.join(os.path.join(method_save_folder, param_str), 'predictor.pt')))
        start_time = time.time()
        network, results = domain_training.training(encoder=ft_encoder,
                                           classifier=classifier,
                                           s_dataloader=train_labeled_ccle,
                                           t_dataloader=labeled_tcga,
                                           drug=drug,
                                           task_save_folder=task_save_folder,
                                           params_str=param_str,
                                           **wrap_params(training_params, type='labeled'))
        results = pd.DataFrame(results)
        elapsed = time.time() - start_time
        print('8-drug Elapsed time: ', round(elapsed, 4))
        with open(f'results/{fold}/{param_str}', 'w') as f:
            results.to_csv(f)
            
        fold = fold+1
        
    with open("{}_auroc.txt".format(param_str), 'a') as opf:
        opf.write('\n')
        opf.write('\n')
    opf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining and Fine_tuning')
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--a_thres', dest='a_thres', nargs='?', type=float, default=None)
    parser.add_argument('--d_thres', dest='days_thres', nargs='?', type=float, default=None)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)
    
    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()
    params_grid = {
        "pretrain_num_epochs": [300],
    }

    keys, values = zip(*params_grid.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param in params_list:
        
        main(args=args, params_dict=param,
             # drug=["Cisplatin", "Paclitaxel", "Cyclophosphamide", "Doxorubicin", "5-Fluorouracil"])
             drug=["Cisplatin", "Paclitaxel", "Cyclophosphamide", "Doxorubicin", "5-Fluorouracil", "Gemcitabine",
                   "Docetaxel", "Etoposide"])
