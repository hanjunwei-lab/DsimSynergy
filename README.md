# DSimSynergy: An Interpretable Framework for Drug Synergy Prediction by integrating Functional and Clinical Drug Similarities
## 1.Introduction
This repository contains source code and data for **DSimSynergy** <br>
**DSimSynergy** is a deep graph learning framework for predicting drug synergy in cancer cell lines. **DSimSynergy** constructs drug-drug similarity networks from biological functional and clinical application information, utilizes molecular fingerprints as initial embedding features of network nodes, and then leverages the graph convolutional networks to learn neighbor information from the similarity networks for optimizing drug representations. It subsequently encodes drug molecular structural features through graph attention networks and processes cell line gene expression data using fully connected neural networks. Finally, the learned embedding features based on drug similarity, molecular structure, and cell line are concatenated and integrated to predict the synergistic effect score for each drug-drug-cell line triplet.<br> 
Comprehensive benchmarking on multiple independent datasets demonstrates that **DSimSynergy** consistently outperforms state-of-the-art methods. Model interpretability analysis revealed key genes and pathways underlying drug synergy, while validation on clinical patient and cohort data demonstrated good clinical translational potential and discovered the molecular mechanisms by which drugs generate synergistic effects through "pathway complementary networks."<br>
## 2.Design of DSimSynergy
<p align="center">
  <img src="image/Overall_architecture.jpg" />
</p>
Figure 1: Overall architecture of DSimSynergy

## 3.Overview
The repository is organised as follows:<br>
- `Code/Data` contains data files and data processing files;
- `Code/Drug Sim Network` contains code for constructing the drug similarity network;
- `Code/Model Classification` contains different modules of **DSimSynergy** classification model;
- `Code/Model Regression` contains different modules of **DSimSynergy** regression model.

## 4.Installation
**DSimSynergy** relies on R (version 4.3.1) and Python (version 3.12.9) environments.<br>
- Install the necessary R packages for **DSimSynergy**:<br>
```
install.packages("tidyr")
install.packages ("dplyr")
```
- Install the necessary python packages for **DSimSynergy**:<br>
```sh
pip install -r requirements.txt
```
| Package         | Version                 |
|:----------------|:------------------------|
| numpy           | 2.12.0                  |
| pandas          | 2.2.3                   |
| scipy           | 1.15.2                  |
| scikit-learn    | 1.6.1                   |
| torch           | 2.8.0.dev20250414+cu128 |
| torch-geometric | 2.6.1                   |
| rdkit           | 2024.9.6                |
| deepchem        | 2.5.0                   |
| shap            | 0.47.2                  |
| matplotlib      | 2.10.0                  |
| seaborn         | 0.13.2                  |

## 5.Usage
The implemention of **DSimSynergy** framework is divided into two main parts:<br>
- **Part I:** Construct two drug similarity networks through biological processes and ATC codes.<br>
- **Part Ⅱ:** The complete deep learning training phase of **DSimSynergy**.<br>
### 5.1.Implemention of Part I
**1. Code** The code used to implement **Part I** is in the file `DSimSynergy_deal.R`,which located at folder `Code/Drug Sim Network`.This file encompasses the original data processing, construct two drug similarity networks.<br>
**2. Data** The datasets used to implement **Part I** are located at folder `Code/Drug Sim Network/Data`
| File                      | Description                                                                         |
|:--------------------------|:------------------------|
| cell_Line.csv             | The correspondence between cell line names and COSMIC ID in the DrugCombDB database.|
| drug_atc.csv              | Information on the ACT code of drugs.                                               |
| drug_smiles.csv           | Information on the smiles coding of drugs.                                          |
| drug_target_9606.csv      | Information on the targets coding of drugs.                                         |
| drugcombdb_drug to db.csv | The correspondence between DrugcombDB drug names and Drugbank ID.                   |
| drugcombs_scored.csv      | The Loewe value of the DrugCombDB triplet.                                          |
| Gosets_BP.csv             | Gene Set for Biological Processes.                                                  |
| landmaker_info.txt        | L1000 Landmark Genes.                                                               |
| Model.csv                 | Cell Line Name and ID Mapping Relationships.                                        |
| OmicsExpressionProteinCodingGenesTPMLogp1.csv  | Gene Expression Values of Cell Lines.                          |

### 5.2.Implemention of Part Ⅱ
**1. Code** The complete training code for **DSimSynergy** is in the file `main.py` which located at folder `Code/Model Classification` and `Code/Model Regression`<br>
Run the following commands to train **DSimSynergy**.<br> 
``` 
python main.py
```
**2. Data** The datasets used to train **DSimSynergy** are located at folder `Data/`
- `Code/Data/TRAIN` contains training set data;
- `Code/Data/DRUGCOOMB` contains external test set data of DrugComb;
- `Code/Data/AZ_DREAM` contains external test set data of AZ_DREAM;
