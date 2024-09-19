# M2GATE

## What is M2GATE?

![Figure1_overview](https://github.com/LYxiaotai/M2GATE/blob/main/Figure1_overview.jpg)

M2GATE, a Graph ATtention auto-Encoder framework featuring two Masking strategies: masked feature reconstruction and re-mask decoding. 

* The graph attention module in M2GATE enables adaptive learning of local spatial neighbors, while the masking strategies enhance model robustness, resulting in stable latent representations with more core biological information. 

* M2GATE adopts an edge removal strategy to construct an enhanced spatial graph, which is more specific for the domain identification task, facilitating clearer delineation of domain boundaries. 

* By employing mask strategies and integrating gene expression data with the enhanced spatial graph, M2GATE learns stable latent representations that improve spatial domain identification and downstream analyses, such as tissue structure visualization, Uniform Manifold Approximation and Projection (UMAP) visualization, spatial trajectory inference, pseudo-spatiotemporal map (pSM) analysis and discovery of domain-specific marker genes.


## How to use M2GATE?

### 1. Requirements

M2GATE is implemented in the pytorch framework (tested on Python 3.9.19). We recommend that users run M2GATE on CUDA. The following packages are required to be able to run everything in this repository (included are the versions we used):

``` python

numpy==1.26.4
pandas==2.2.2
sklearn==1.5.0
scipy==1.13.1
tqdm==4.66.4
scanpy==1.10.1
anndata==0.10.7
torch==2.3.1+cu121
palettable==3.3.3
ryp2==3.5.16
R==4.2.2

```


### 2. Tutorial

(1) Download the [M2GATE_F.py](https://github.com/LYxiaotai/M2GATE/blob/main) and [gat_conv2.py](https://github.com/LYxiaotai/M2GATE/blob/main)

(2) Tutorial for domain identification on 10x Visium human dorsolateral prefrontal cortex (DLPFC) dataset can be found here: [Domian_151674.ipynb](https://github.com/LYxiaotai/M2GATE/blob/main)

* The datasets for [Tutorial](https://github.com/LYxiaotai/M2GATE/blob/main/Domian_151674.ipynb) can be found [here](https://github.com/LYxiaotai/M2GATE/blob/main/151674)
* All the results of [Tutorial](https://github.com/LYxiaotai/M2GATE/blob/main/Domian_151674.ipynb) can be found [here](https://github.com/LYxiaotai/M2GATE/blob/main/151674/results)



