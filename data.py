import gzip
import random
import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs, Chem
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from torch.utils.data import TensorDataset, DataLoader
import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_unlabeled_dataloaders(gex_features_df, seed, batch_size, ccle_only=False):
    """CCLE as source domain, tcga(TCGA) as target domain"""
    set_seed(seed)
    ccle_sample_info_df = pd.read_csv(config.ccle_sample_file, index_col=0)
    with gzip.open(config.tcga_sample_file) as f:
        tcga_sample_info_df = pd.read_csv(f, sep='\t', index_col=0)
    tcga_samples = tcga_sample_info_df.index.intersection(gex_features_df.index)
    ccle_samples = gex_features_df.index.difference(tcga_samples)
    tcga_sample_info_df = tcga_sample_info_df.loc[tcga_samples]
    ccle_sample_info_df = ccle_sample_info_df.loc[ccle_samples.intersection(ccle_sample_info_df.index)]

    tcga_df = gex_features_df.loc[tcga_samples]
    ccle_df = gex_features_df.loc[ccle_samples]

    excluded_ccle_samples = []
    excluded_ccle_samples.extend(ccle_df.index.difference(ccle_sample_info_df.index))
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < 2].index
    excluded_ccle_samples.extend(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)

    to_split_ccle_df = ccle_df[~ccle_df.index.isin(excluded_ccle_samples)]
    train_ccle_df, test_ccle_df = train_test_split(to_split_ccle_df, test_size=0.1,
                                                   stratify=ccle_sample_info_df.loc[to_split_ccle_df.index].primary_disease)
    test_ccle_df = pd.concat([test_ccle_df, ccle_df.loc[excluded_ccle_samples]])
    #test_ccle_df = test_ccle_df.append(ccle_df.loc[excluded_ccle_samples])
    train_tcga_df, test_tcga_df = train_test_split(tcga_df, test_size=len(test_ccle_df) / len(tcga_df),
                                                   stratify=tcga_sample_info_df['_primary_disease'], random_state=seed)

    tcga_dataset = TensorDataset(torch.from_numpy(tcga_df.values.astype('float32')))
    ccle_dataset = TensorDataset(torch.from_numpy(ccle_df.values.astype('float32')))
    test_tcga_dateset = TensorDataset(torch.from_numpy(test_tcga_df.values.astype('float32')))
    test_ccle_dateset = TensorDataset(torch.from_numpy(test_ccle_df.values.astype('float32')))

    tcga_dataloader = DataLoader(tcga_dataset, batch_size=batch_size, shuffle=True)
    test_tcga_dataloader = DataLoader(test_tcga_dateset, batch_size=batch_size, shuffle=True)
    ccle_data_loader = DataLoader(ccle_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_ccle_dataloader = DataLoader(test_ccle_dateset, batch_size=batch_size, shuffle=True)

    if ccle_only:
        return (ccle_data_loader, test_ccle_dataloader), (ccle_data_loader, test_ccle_dataloader)
    else:
        return (ccle_data_loader, test_ccle_dataloader), (tcga_dataloader, test_tcga_dataloader)
    

def get_tcga_multi_labeled_dataloaders(gex_features_df, drug, batch_size, days_threshold_list=None, nan_flag=False, tcga_cancer_type=None):
    if tcga_cancer_type is not None:
        raise NotImplementedError("Only support pan-cancer")

    drugs_to_keep = [item.lower() for item in drug]

    # Filter by beginning string "TCGA"
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    # Take the first 12 characters of the original string as the new data id.
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    # Group by the new id and get the average of each column as the features.
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()

    tcga_labeled_df = pd.read_csv(config.tcga_multi_label_file_8, header=0, index_col=0)
    intersection_index = tcga_gex_feature_df.index.intersection(tcga_labeled_df.index)

    tcga_labeled_df = tcga_labeled_df.loc[intersection_index]
    tcga_labeled_gex_feature_df = tcga_gex_feature_df.loc[intersection_index]
    print("tcga_labeled_df========================================================")
    print(tcga_labeled_df)
    if nan_flag == False:
        tcga_labeled_df = tcga_labeled_df.replace(0.0, -1.0)
        tcga_labeled_df = tcga_labeled_df.replace(np.nan, 0)
    print("replaced_tcga_labeled_df========================================================")
    print(tcga_labeled_df)

    print("tcga_labeled_gex_feature_df==============================================")
    print(tcga_labeled_gex_feature_df)

    labeled_tcga_dateset = TensorDataset(
        torch.from_numpy(tcga_labeled_gex_feature_df.values.astype('float32')),
        torch.from_numpy(tcga_labeled_df.values.astype('float32')))

    labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                         batch_size=batch_size,
                                         shuffle=False)

    return labeled_tcga_dataloader


def get_ccle_multi_labeled_dataloader_generator(gex_features_df, batch_size, drug, seed=2023, threshold_list=None,
                                          measurement='AUC', nan_flag=False, n_splits=5):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [item.lower() for item in drug]
    # print(drugs_to_keep)

    # read file
    gdsc1_response = pd.read_csv(config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(config.gdsc_target_file2)

    # filter data columns, only get three columns COSMIC_ID, DRUG_NAME and measurement(IC50,AUC,RMSE,Z_SCORE)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    # filter drugs
    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]

    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()

    # Delete the duplicates in gdsc1_target_df and gdsc2_target_df
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    # Take the "COSMIC_ID" as the new row index and take the "DRUG_NAME" as the column index
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')

    # read ccle sample info, get the "COSMICID" column
    ccle_sample_info = pd.read_csv(config.ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    # read gdsc sample info, get the "COSMIC identifier" column
    gdsc_sample_info = pd.read_csv(config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]

    # inner join with index
    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    # the index is the COSMIC_ID and the value is the DepMap_ID
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']
    # change the data index, the final index is the DepMap_ID.
    # Values with no mapping relationship are set to NaN.
    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]

    ccle_target_df = target_df[drugs_to_keep]
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    ccle_labels = ccle_target_df.loc[ccle_labeled_samples]
    # print(ccle_labels)
    for i in range(len(ccle_labels)):
        for j in range(len(drugs_to_keep)):
            if np.isnan(ccle_labels.iloc[i, j]):
                continue
            else:
                ccle_labels.iloc[i, j] = (ccle_labels.iloc[i, j] < threshold).astype(int)
    print("Cell Line labels==============================================================")
    print(ccle_labels)
    if nan_flag == False:
        ccle_labels = ccle_labels.replace(0.0, -1.0)
        ccle_labels = ccle_labels.replace(np.nan, 0)
    print("Replaced ccle_labels=====================================================")
    print(ccle_labels)

    print("Sum of Cell Line labels=======================================================")
    print(ccle_labels.sum())
    print(ccle_labels.sum().values.tolist())

    config.label_graph = np.eye(len(drug), dtype=float)
    
    drug_id_map = dict()
    for idx, item in enumerate(drug):
        drug_id_map[item] = idx
    
    for idx, item in enumerate(drug):
        occurence_dict = dict()
        for i in range(len(ccle_labels)):
            if ccle_labels.iloc[i][idx] == 1:
                for j in range(len(ccle_labels.iloc[i])):
                    if j == idx:
                        continue
                    if ccle_labels.iloc[i][j] == 1:
                        if drug[j] in occurence_dict.keys():
                            occurence_dict[drug[j]] = occurence_dict[drug[j]] + 1
                        else:
                            occurence_dict[drug[j]] = 1
        occurence_rank = sorted(occurence_dict.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(len(occurence_rank)):
            config.label_graph[idx][drug_id_map[occurence_rank[i][0]]] = occurence_rank[i][1]
            
    row, col = np.diag_indices_from(config.label_graph)
    config.label_graph[row, col] = np.array(ccle_labels.sum().values.tolist())
    
    print("Label Graph===================================================================")
    print(config.label_graph)
    config.label_graph_norm = config.label_graph
    for idx_col in range(config.label_graph_norm.shape[1]):
        normalizer = np.sum(config.label_graph_norm[:, idx_col])
        config.label_graph_norm[:, idx_col] = config.label_graph_norm[:, idx_col] * 1.0 / normalizer
    
    print("Normalizered Label Graph============================================================")
    threshold_label = 0.1
    config.label_graph_diag = config.label_graph_norm - np.diag(np.diag(config.label_graph_norm))
    config.label_graph_diag = (config.label_graph_diag >= threshold_label).astype(int)
    print(config.label_graph_diag)
    
    print("Label Similarity============================================================")
    mol_list = [Chem.MolFromSmiles(x) for x in config.smiles_list]
    fp_list = [Chem.RDKFingerprint(x) for x in mol_list]
    drug_similarity = []
    for i in range(len(mol_list)):
        sim_list = []
        for j in range(len(mol_list)):
            val = DataStructs.FingerprintSimilarity(fp_list[i], fp_list[j])
            sim_list.append(val)
        drug_similarity.append(sim_list)
    config.drug_similarity = np.array(drug_similarity)
    print(config.drug_similarity)

    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]

    kfold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_index, test_index in kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df, = ccle_labeled_feature_df.values[train_index], \
                                                       ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]
        
        train_labeled_ccle_dateset = TensorDataset(
            torch.from_numpy(train_labeled_ccle_df.astype('float32')),
            torch.from_numpy(train_ccle_labels))
        test_labeled_ccle_df = TensorDataset(
            torch.from_numpy(test_labeled_ccle_df.astype('float32')),
            torch.from_numpy(test_ccle_labels))

        train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_df,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader


def get_ccle_labeled_dataloader_generator(gex_features_df, drug, batch_size, seed=2020, threshold=None,
                                          measurement='AUC', n_splits=5):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc1_response = pd.read_csv(config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(config.gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
    ccle_sample_info = pd.read_csv(config.ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    gdsc_sample_info = pd.read_csv(config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]

    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']

    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]

    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    s_kfold = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]

        train_labeled_ccle_dateset = TensorDataset(
            torch.from_numpy(train_labeled_ccle_df.astype('float32')),
            torch.from_numpy(train_ccle_labels))
        test_labeled_ccle_df = TensorDataset(
            torch.from_numpy(test_labeled_ccle_df.astype('float32')),
            torch.from_numpy(test_ccle_labels))

        train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_df,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader



def get_multi_labeled_dataloader_generator(gex_features_df, drug, seed, batch_size, ccle_measurement='AUC',
                                       threshold_list=None,
                                       days_threshold_list=None,
                                       n_splits=5):
    ccle_labeled_dataloader_generator = get_ccle_multi_labeled_dataloader_generator(gex_features_df=gex_features_df,
                                                                              batch_size=batch_size,
                                                                              drug=drug,
                                                                              seed=seed,
                                                                              threshold_list=threshold_list,
                                                                              measurement=ccle_measurement,
                                                                              n_splits=n_splits)

    test_labeled_dataloader = get_tcga_multi_labeled_dataloaders(gex_features_df=gex_features_df,
                                                           batch_size=batch_size,
                                                           drug=drug,
                                                           days_threshold_list=days_threshold_list,
                                                           )

    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader in ccle_labeled_dataloader_generator:
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, test_labeled_dataloader
