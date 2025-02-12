import os

import numpy

"""
configuration file includes all related datasets 
"""

root_data_folder = 'data/'
raw_data_folder = os.path.join(root_data_folder, 'raw_dat')
preprocessed_data_folder = os.path.join(root_data_folder, 'preprocessed_dat')
gene_feature_file = os.path.join(preprocessed_data_folder, 'CosmicHGNC_list.tsv')
gdsc_tcga_mapping_file = os.path.join(root_data_folder, 'tcga_gdsc_drug_mapping.csv')

#TCGA_datasets
tcga_folder = os.path.join(root_data_folder, 'tcga')
tcga_clinical_folder = os.path.join(tcga_folder, 'Clinical')
tcga_drug_name_mapping_file = os.path.join(tcga_folder, 'drug_name_mapping.csv')
tcga_first_treatment_file = os.path.join(tcga_folder, 'tcga_drug_first_treatment.csv')
tcga_first_response_file = os.path.join(tcga_folder, 'tcga_drug_first_response.csv')
#tcga_first_response_file = os.path.join(tcga_folder, 'tcga_drug_first_response_type.csv')
tcga_multi_label_file_5 = os.path.join(root_data_folder, 'patient-5drugs.csv')
tcga_multi_label_file_8 = os.path.join(root_data_folder, 'patient-8drugs.csv')


#Xena datasets
xena_folder = os.path.join(raw_data_folder, 'Xena')
xena_id_mapping_file = os.path.join(xena_folder, 'gencode.v23.annotation.gene.probemap')
xena_gex_file = os.path.join(xena_folder, 'tcga_RSEM_gene_tpm.gz')
xena_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'xena_gex')
xena_sample_file = os.path.join(xena_folder, 'TCGA_phenotype_denseDataOnlyDownload.tsv.gz')

#CCLE datasets
ccle_folder = os.path.join(raw_data_folder, 'CCLE')
ccle_gex_file = os.path.join(ccle_folder, 'CCLE_expression.csv')
ccle_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'ccle_gex')
ccle_sample_file = os.path.join(ccle_folder, 'sample_info.csv')

#gex features
gex_feature_file = os.path.join(preprocessed_data_folder, 'tcga_ccle_1000_gex_features.csv')

#GDSC datasets
gdsc_folder = os.path.join(raw_data_folder, 'GDSC')
gdsc_target_file1 = os.path.join(gdsc_folder, 'GDSC1_fitted_dose_response_25Feb20.csv')
gdsc_target_file2 = os.path.join(gdsc_folder, 'GDSC2_fitted_dose_response_25Feb20.csv')
gdsc_raw_target_file = os.path.join(gdsc_folder, 'gdsc_ic50flag.csv')
gdsc_sample_file = os.path.join(gdsc_folder, 'gdsc_cell_line_annotation.csv')
gdsc_preprocessed_target_file = os.path.join(preprocessed_data_folder, 'gdsc_ic50flag.csv')

#adae brain cancer datasets
adae_folder = os.path.join(root_data_folder, 'adae_data')
adae_gex_file = os.path.join(adae_folder, 'TCGA_GBM_and_LGG_PREPROCESSED_RNASEQ_EXPRESSION_500_kmeans.tsv')
adae_sex_label_file = os.path.join(adae_folder, 'TCGA_GBM_and_LGG_SEX_LABELS.tsv')
adae_subtype_label_file = os.path.join(adae_folder, 'TCGA_GBM_and_LGG_SUBTYPE_LABELS.tsv')

#PDTC datasets
pdtc_folder = os.path.join(root_data_folder, 'PDTC')
gdsc_pdtc_drug_name_mapping_file = os.path.join(root_data_folder, 'pdtc_gdsc_drug_mapping.csv')
pdtc_gex_file = os.path.join(preprocessed_data_folder, 'pdtc_uq1000_feature.csv')
pdtc_target_file = os.path.join(pdtc_folder, 'DrugResponsesAUCModels.txt')

#Celligner datasets
celligner_folder = os.path.join(root_data_folder, 'celligner')
celligner_pdtc_gex_file = os.path.join(preprocessed_data_folder, 'celligner_pdtc_uq_df.csv')
celligner_xena_gex_file = os.path.join(preprocessed_data_folder, 'celligner_xena_uq_df.csv')

label_graph = numpy.array(object=object)
label_graph_norm = numpy.array(object=object)
label_graph_norm_empty_diag = numpy.array(object=object)

smiles_list = ['N.N.Cl[Pt]Cl', 'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C',
               'C1CNP(=O)(OC1)N(CCCl)CCCl', 'CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O',
               'C1=C(C(=O)NC(=O)N1)F', 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F', 'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)OC(C)(C)C)O)O)OC(=O)C6=CC=CC=C6)(CO4)OC(=O)C)O)C)O',
               'CC1OCC2C(O1)C(C(C(O2)OC3C4COC(=O)C4C(C5=CC6=C(C=C35)OCO6)C7=CC(=C(C(=C7)OC)O)OC)O)O']