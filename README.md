# AdaptCDR
A Multi-Label Heuristic Domain Adaptation Model for Transferring Cancer Drug Response Prediction

![Framework of AdaptCDR](https://github.com/luodandan/AdaptCDR/blob/main/Framework.png)  


## Requirements
* Python >= 3.7
* PyTorch >= 1.5
* PyTorch Geometry >= 1.6
* hickle >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09
  
## Overview 
AdaptCDR begins by pre-training an autoencoder to extract shared genomic features across domains, and a multi-label classifier (as a decoder) to predict response outcomes for multiple drugs in the source domain. One unique advantage of our multi-label classifier is the incorporation of an association graph, where each drug/label is treated as a node, allowing dynamic learning of inter-label associations (edges). Then, a heuristic domain adversarial network is introduced to bridge genomic disparity between two domains, where the domain-invariant representation that leads to larger domain discrepancy is identified as the domain-specific counterpart.

## Installation
1. Install anaconda:
Instructions here: https://www.anaconda.com/download/
2. pip install -r Requirements.txt
3. The data can be downloaded from here：IEEE dataport DOI：10.21227/e1xk-2r37
4. Run main.py

