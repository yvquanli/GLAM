# GLAM
Code for "An adaptive graph learning method for automated molecular interactions and properties predictions".

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) 
[![Paper](https://img.shields.io/badge/paper-Nature%20Machine%20Intelligence-green)](https://www.nature.com/articles/s42256-022-00501-8)
![version](https://img.shields.io/badge/version-v3.0.0-blue) 

Access the paper by this sharedit link: https://rdcu.be/cQP0g

## Abstract

Improving drug discovery efficiency is a core and long-standing challenge in drug discovery. For this purpose, many graph learning methods have been developed to search potential drug candidates with fast speed and low cost. In fact, the pursuit of high prediction performance on a limited number of datasets has crystallized their architectures and hyperparameters, making them lose advantage in repurposing to new data generated in drug discovery. Here we propose a flexible method that can adapt to any dataset and make accurate predictions. The proposed method employs an adaptive pipeline to learn from a dataset and output a predictor. Without any manual intervention, the method achieves far better prediction performance on all tested datasets than traditional methods, which are based on hand-designed neural architectures and other fixed items. In addition, we found that the proposed method is more robust than traditional methods and can provide meaningful interpretability. Given the above, the proposed method can serve as a reliable method to predict molecular interactions and properties with high adaptability, performance, robustness and interpretability. This work would take a solid step forward to the purpose of aiding researchers to design better drugs with high efficiency.


## Requirements
Our work is implementated based on version 1.7.2 of pyg. 

    conda >= 4.9.2
    PyTorch >= 1.5.0
    torch-gemetric == 1.7.2
    rdkit >= '2019.03.4'

## Installation
First You should choose the Anaconda version that suits your system and install it by:

    wget https://repo.anaconda.com/archive/Anaconda3-2021.04-Linux-x86_64.sh
    sh Anaconda3-2021.04-Linux-x86_64.sh
    # wget https://repo.anaconda.com/archive/Anaconda3-2021.04-MacOSX-x86_64.sh
    # sh Anaconda3-2021.04-MacOSX-x86_64.sh

You can install the required dependencies with the following code. 
    
    conda create -n GLAM python=3.8 rdkit pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge --yes
    conda activate pyg_old
    CUDA=cu111
    TORCH=1.9.0
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html 
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html 
    pip install torch-geometric==1.7.2 

If you don't have a gpu or want a cpu version, you can try this:
    
    conda create -n GLAM python=3.8 rdkit -c conda-forge --yes
    conda activate GLAM
    conda install pytorch=1.9.0 cpuonly -c pytorch -c conda-forge --yes
    CUDA=cpu
    TORCH=1.9.0
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html 
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html 
    pip install torch-geometric==1.7.2 
    git clone https://github.com/yvquanli/GLAM.git



## Demo
If you have successfully completed the above steps, then you can use the following code to run a demo for a property prediction task.

    cd ./GLAM/src_1gp
    python3 demo.py


## Dataset
All datasets can be download from these website:

- LIT-PCBA(ALDH1, ESR1_ant, KAT2A, MAPK1): http://drugdesign.unistra.fr/LIT-PCBA/
- BindingDB: https://github.com/lifanchen-simm/transformerCPI/blob/master/data/BindingDB.zip
- DrugBank: https://github.com/kexinhuang12345/CASTER/blob/master/DDE/data/unsup_dataset.csv
- MoleculeNet(ESOL, Lipophilicity, FreeSolv, BACE, BBBP, SIDER, Tox21, ToxCast): http://moleculenet.org/datasets-1
- Perturbed dataset built from PhysProp: here ./physprop_perturb.csv

## Full structure of workplace
If you have all the datasets and code ready, you should place these files in the following structure.

    .
    ├── Dataset/  
    │   ├── GLAM-DDI/  
    │   │   └── raw/  
    │   │       └── drugbank_caster/
    │   │           └── ddi_total.csv
    │   ├── GLAM-DTI/
    │   │   └── raw/
    │   │       ├── bindingdb/
    │   │       │   ├── pro_contact_map/protein_maps_dict.ckpt
    │   │       │   ├── train.txt
    │   │       │   ├── dev.txt
    │   │       │   └── test.txt
    │   │       └── lit_pcba_raw/
    │   │           ├── raptor_pred/contact_8.5/protein_maps_dict.ckpt
    │   │           ├── ALDH1/
    │   │           └── ...
    │   └── GLAM-GP/
    │       └── raw/
    │           ├── bace.csv
    │           └── ...
    └── GLAM/
        ├── LICENSE
        ├── README.md
        ├── src_1gp/
        │   └── ...
        ├── src_2gi_ddi/
        │   └── ...
        └── src_2gi_dti_scr/
            └── ...



- ./GLAM/src_1gp: The source code for molecular property prediction.
- ./GLAM/src_2gi_ddi: The source code for molecular pair interaction identification of drug-drug task.
- ./GLAM/src_2gi_dti_scr: The source code for molecular pair interaction identification of drug-target and screening task.
- ./Dataset/GLAM-DTI/raw: The raw data must be placed here.
- ./Dataset/GLAM-DTI/processed: The processed data will be stored here
- etc...




## Usage

Then make sure that run.py can be run and done by

    python3 run.py --epochs 1

Then run glam.py by

    python3 glam.py [-h] [--dataset DATASET] [--n_init_configs N_INIT_CONFIGS]
                     [--n_low_fidelity_seed n_low_fidelity_seed]
                     [--n_top_blend N_TOP_BLEND]
                     [--n_high_fidelity_seed n_high_fidelity_seed]

optional arguments:

      -h, --help            show this help message and exit
      --dataset DATASET     bindingdb_c, lit_esr1ant
      --n_init_configs N_INIT_CONFIGS
                            n initialized configurations
      --n_low_fidelity_seed n_low_fidelity_seed
                            n run for few epochs
      --n_top_blend N_TOP_BLEND
                            auto blend n models
      --n_high_fidelity_seed n_high_fidelity_seed
                            n run for full epochs with a config

If you want to perform property-slightly-affected structural perturbation with GLAM:

    python3 glam.py --dataset physprop_perturb
    
If you want to perform property-slightly-affected structural perturbation with your model, please feed your model and physprop dataset to TrainerMolRegression and use the pasp() function.
    

# Citation

Li, Y., Hsieh, CY., Lu, R. et al. An adaptive graph learning method for automated molecular interactions and properties predictions. Nat Mach Intell (2022). https://doi.org/10.1038/s42256-022-00501-8
