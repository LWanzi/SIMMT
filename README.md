# SIMMT

Image-enhanced Multi-Modal Contrastive Transformer for subcellular spatial transcriptomics

---

## Introduction
We present an image-enhanced multi-modal contrastive transformer framework SIMMT for identifying spatial domains and enhancing subcellular data. In the framework, we design a dual transformer architecture to learn multi-modal representations for cells by modeling transcriptomics and morphological images respectively. To fully capture modality interactions within spatial contexts, we introduce a contrastive learning module that enhances cell representation by aligning tissue morphology and gene expression at the cell level. 


---

## Requirements

- Python == 3.8  
- PyTorch == 1.10.0  
- h5py == 3.11.0  
- scanpy == 1.9.8 
- numpy == 1.24.4  
- opencv-python==4.11.0
- umap-learn == 0.5.3  
- pandas==2.0.3
- torch-geometric==2.6.1
- scikit-learn ==1.3.2 

---

## Usage

Run SIMMT from the command line:

```bash
# Training on CosMx dataset with default settings
python train.py --dataset nanostring --epochs 900 --root ../dataset/nanostring --ncluster 8

# Training on 10x dataset
python train.py --dataset 10x --epochs 900 --root ../dataset/HBA --ncluster 4
```

### Key arguments

| Parameter          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `--dataset`        | Dataset type, choose from `nanostring` or `10x`.                            |
| `--lr`             | Learning rate (default: `1e-3`).                                            |
| `--root`           | Path to dataset root directory.                                             |
| `--epochs`         | Number of training epochs (default: `900`).                                 |
| `--seed`           | Random seed (default: `1234`).                                              |
| `--save_path`      | Directory to save checkpoints and results.                                   |
| `--ncluster`       | Number of clusters (default: `8`).                                          |
| `--repeat`         | Number of repeats for clustering (default: `1`).                            |
| `--use_gray`       | Use grayscale images (default: `0`).                                        |
| `--test_only`      | Set to `1` for test-only mode (default: `0` for training).                  |
| `--pretrain`       | Path to pre-trained model (default: `all/final_900_0.pth`).                 |
| `--cluster_method` | Clustering algorithm: `leiden` or `mclust`.                                 |


