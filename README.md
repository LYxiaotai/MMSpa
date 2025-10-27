# MMSpa

## What is MMSpa?

![Figure1_overview.png](https://github.com/LYxiaotai/MMSpa/blob/main/Figure1_overview.png)

MMSpa, a Graph ATtention auto-Encoder framework featuring two Masking strategies: masked feature reconstruction and re-mask decoding. 

* The graph attention module in MMSpa enables adaptive learning of local spatial neighbors, while the masking strategies enhance model robustness, resulting in stable latent representations with more core biological information. 

* MMSpa adopts an edge removal strategy to construct an enhanced spatial graph, which is more specific for the domain identification task, facilitating clearer delineation of domain boundaries. 

* By employing mask strategies and integrating gene expression data with the enhanced spatial graph, M2GATE learns stable latent representations that improve spatial domain identification and downstream analyses, such as tissue structure visualization, Uniform Manifold Approximation and Projection (UMAP) visualization, spatial trajectory inference, pseudo-spatiotemporal map (pSM) analysis and discovery of domain-specific marker genes.


## How to use MMSpa?

### 1. Requirements
  
MMSpa is implemented in the PyTorch framework (tested on Python 3.9.19). We recommend that users run MMSpa on CUDA.

- `Python 3.9`

MMSpa also calls the "mclust" package (tested on mclust==6.0.1) in the R language.

- `R (We recommend using the R Version 4.2.2.)`

The use of the mclust algorithm requires the rpy2 package and the mclust package. See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for details.

The rpy2 package is included in requirement.txt. Installation: see Step 2.

The installation of the mclust package can be found in Step 3.

#### Quick Installation：
#### Step 1: Before installing the Python dependencies, we recommend that users create a new conda environment and R environment.
-   `Conda create -n env python=3.9`
-   `conda install r-base`
#### Step 2: Install Python dependencies
-   `pip install -r requirement.txt`
#### Step 3: Install R dependencies (requires R environment, see Step 1)
-   `conda install r-mclust`


### 2. Tutorial

(1) Download the [MMSpa_F.py](https://github.com/LYxiaotai/MMSpa/blob/main) and [gat_conv2.py](https://github.com/LYxiaotai/MMSpa/blob/main).

(2) Tutorial for domain identification on 10x Visium human dorsolateral prefrontal cortex (DLPFC) dataset can be found here: [Train_151674.ipynb](https://github.com/LYxiaotai/MMSpa/blob/main/Train_151674.ipynb).

* The datasets for the Tutorial [Train_151674.ipynb](https://github.com/LYxiaotai/MMSpa/blob/main/Train_151674.ipynb) can be found [here](https://github.com/LYxiaotai/MMSpa/tree/main/data/151674).
* The domain identification results of the Tutorial [Train_151674.ipynb](https://github.com/LYxiaotai/MMSpa/blob/main/Train_151674.ipynb) can be found [here](https://github.com/LYxiaotai/MMSpa/tree/main/data/results).
* When facing to determining the number of clusters without prior knowledge, we recommend selecting the number of clusters by identifying the maximum score within a range of potential values, using metrics such as the Silhouette score. We have provided the function `cluster_select()` to help select the highest Silhouette score in [MMSpa_F.py](https://github.com/LYxiaotai/MMSpa/blob/main/MMSpa_F.py).
 ```
 # A simple example:

  min_clust = 20
  max_clust = 25
  figsave = './MMSpa/'
  cluster_sil = MMSpa_F.cluster_select(adata, min_clust, max_clust, figsave) 
  optimal_clusters = max(cluster_sil, key=cluster_sil.get)    # find optimal cluster number

# cluster_sil：A dictionary containing silhouette scores for each tested cluster number. Example: {3: 0.723, 4: 0.815, 5: 0.782}
```
* Additionally, recent advances in the field of statistics, nonparametric Bayesian, such as the nonparametric Potts prior, offer inferring spatial domains in a fully data-driven manner. A related reference is Yan and Luo (2024, Bayesian integrative region segmentation in spatially resolved transcriptomic studies, JASA) (https://doi.org/10.1080/01621459.2024.2308323).



