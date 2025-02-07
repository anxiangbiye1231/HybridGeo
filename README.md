# HybridGeo
---
Geometric Deep Learning for Protein-Ligand Affinity Prediction with Hybrid Message Passing Strategies

## Datasets
All data used in this paper are publicly available and can be accessed here:  
- PDBbind v2020: http://www.pdbbind.org.cn/download.php  
- 2013 and 2016 core sets: http://www.pdbbind.org.cn/casf.php  
You can find processed data from `./data`, and change the path to run.

## Note 

This project contains several GNN-based models for protein-ligand binding affinity prediction, which are mainly taken from  
- PotentialNet: https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/potentialne  
- GNN_DTI: https://github.com/jaechanglim/GNN_DTI  
- IGN: https://github.com/zjujdj/InteractionGraphNet/tree/master  
- SchNet: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html  
- GraphDTA: https://github.com/thinng/GraphDTA   
- MGraphGTA: https://github.com/guaguabujianle/MGraphDTA   
- MMDTA: https://github.com/dldxzx/MMDTA
- EGNN：https://github.com/vgsatorras/egnn
- GIGN：https://github.com/guaguabujianle/GIGN  

## Requirements
biopython==1.79  
networkx==3.2.1  
numpy==1.23.5    
pandas==2.2.1    
pymol==3.0.0  
python==3.10.0   
rdkit==2023.9.5    
scikit-learn==1.4.1    
scipy==1.12.0    
torch==2.0.1     
torch-geometric==2.5.2   
tqdm==4.66.2  
## Resource requirements
We tested the training and inference time, along with the resource requirement of HybridGeo on a server computer embedded with a graphical card NVIDIA GeForce RTX4090D with a memory of 24G. HybridGeo takes 105 minutes to train. For the resource requirement, the training of MHAN-DTA costs 13220M of graphics memory.
## Usage
We provide a demo to show how to train and test HybridGeo.   
### 1. Training 
1. Firstly,  follow the data processing steps to generate the train and valid parts of the data.   
2. Secondly, run train.py using `python train.py`.  
### 2. Testing  
1. Unzip the saved models and preprocessed data from `./data/internal_test.tar.gz`, `./data/test_2013.tar.gz`, `./data/test_2016.tar.gz` and `./data/test_hiq.tar.gz`.   
2. Run test.py using `python predict.py`.    
3. You may need to modify some file paths in the source code before running it.    
download the models and data from https://pan.baidu.com/s/1Sn5PbgwtDWx66BKGsclYyA?pwd=q4xm Password：q4xm    
### 3. Process raw data  
1. `preprocessing.py` contains pocket generation, complex construction, and esm feature extraction.       
2. `dataset.py` contains residue geometric feature building.
3. Please debug appropriately according to the data storage directory and the required format.    
