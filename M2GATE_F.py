import random
import numpy as np
import os
import torch
import scanpy as sc
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from gat_conv2 import En_DecoderGAT
import scipy.sparse as sp
import torch.nn as nn
from functools import partial




# set seed
def set_seed(seed):

    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"]= "1"


    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    # torch.use_deterministic_algorithms(True)



# sce loss function
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss




# Train: Symmetrical GAT & Mask
class PreModel1(nn.Module):
    def __init__(self,
                 in_dim,
                 replace_rate,
                 mask_rate,
                 alpha_l,  # sce r
                 en_hidden_channels=256,
                 en_num_layers=2,
                 num_hidden=30,
                 de_hidden_channels=256,
                 de_num_layers=2,  
                 loss_fn='sce',
                 concat_hidden=False):
        
        super(PreModel1, self).__init__()

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._mask_rate = mask_rate
        self._concat_hidden = concat_hidden

        self.encoder = En_DecoderGAT(in_dim, 
                                     en_hidden_channels, 
                                     num_hidden,
                                     en_num_layers)
        
        de_in_channels = num_hidden
        self.decoder = En_DecoderGAT(de_in_channels, 
                                     de_hidden_channels, 
                                     in_dim,
                                     de_num_layers)
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)


    def setup_loss_fn(self, loss_fn, alpha_l):
        
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion


    def encoding_mask_noise(self, g, x, mask_rate):
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)


    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        enc_rep[mask_nodes] = 0

        recon, _ = self.decoder(pre_use_g, enc_rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss, recon


    def forward(self, g, x):
        
        # ---- attribute reconstruction ----
        loss, recon = self.mask_attr_prediction(g, x)
        return loss, recon 


    def embed(self, g, x):
        rep, _ = self.encoder(g, x)
        return rep
     




def Train_M2GATE(model, edge, dataX, adata):
    
    set_seed(11)

    weight_decay = 0
    lr = 0.001
    max_epoch = 2000
    n_epochs = 1000

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)


    Loss = []
    for epoch in tqdm(range(1, 1 + n_epochs)):
        model.train()
        loss,_ = model(edge, dataX)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        Loss.append(loss.item())


    with torch.no_grad():
        model.eval()    
        rep = model.embed(edge, dataX)
        adata.obsm['hidden_re'] = rep.to('cpu').detach().numpy()
        _ , recon = model(edge, dataX)   
        adata.obsm['recon'] = recon.to('cpu').detach().numpy()


    adata.uns['loss'] = Loss
    
    return adata






# read adata
def read_data(path, count_file, datatype):
    
    if datatype == 'h5':
        adata = sc.read_visium(path=path, count_file=count_file)
    if datatype == 'h5ad':
        adata = sc.read_h5ad( path + count_file)

    adata.var_names_make_unique()

    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # print('original adata: ',adata)

    return adata


#  normalize adjacency matrix
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()



def get_data_deladj(adata, k_cutoff, exp_cutoff):

    # highly gene exp
    adata_var = adata[:, adata.var['highly_variable']]
    adataX = np.array(adata_var.X.todense())

    # del_adj
    from sklearn.neighbors import kneighbors_graph
    from scipy.spatial.distance import cdist

    coor = pd.DataFrame(adata.obsm['spatial'])
    corrx = list(coor[0])
    corry = list(coor[1])
    corrxy = []
    corrxy.append(corrx)
    corrxy.append(corry)
    corrxy = np.array(corrxy).T

    locations_metric='minkowski'
    locations_metric_p=2
    num_neighbors_target = k_cutoff + 1
    G_df = kneighbors_graph(corrxy, num_neighbors_target, mode='connectivity', include_self=True,
                                    metric=locations_metric, p=locations_metric_p)
    adj = np.array(G_df.todense())

    distance_matrix= cdist(adataX, adataX,metric='euclidean')
    neighbor_indices = np.argsort(-distance_matrix, axis=1)[:, :exp_cutoff]
    adjacency_matrix = np.zeros_like(distance_matrix, dtype=int)
    adjacency_matrix[np.arange(distance_matrix.shape[0])[:, None], neighbor_indices] = 1

    del_adj = adj - adjacency_matrix
    del_adj = np.where(del_adj < 0, 0, del_adj)
    # del_adj_normal = normalize_adj(del_adj)

    dataX = torch.FloatTensor(adataX)
    edge = torch.FloatTensor(del_adj)     # is_symmetric = torch.allclose(edge, edge.T) = False
    # edge_normal = torch.FloatTensor(del_adj_normal)

    return dataX, edge




# cluster method: mclust
def mclust_R(adata, 
             num_cluster, 
             modelNames='EEE', 
             used_obsm='hidden_re', 
             random_seed=2020):
    
    np.random.seed(random_seed)
    
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)

    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), 
                  num_cluster, 
                  modelNames)

    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata




# cluster method: leiden
def res_search_fixed_clus_leiden(adata, fixed_clus_count, min_res=0.1, max_res=30, increment=0.01):

        
        import scanpy as sc
        import pandas as pd
        import numpy as np

        n = 0
        ranges = list(np.arange(min_res, max_res, increment))
        for res in sorted((ranges), reverse=True):
            n+=1
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == fixed_clus_count:
                break
            if n == len(ranges):
                print('Failed to find the suitable resolution !!! Please change the range !!!')
            
        return res


# cluster method: louvain
def res_search_fixed_clus_louvain(adata, fixed_clus_count, min_res=0.1, max_res=30, increment=0.01):

        
        import scanpy as sc
        import pandas as pd
        import numpy as np


        ranges = list(np.arange(min_res, max_res, increment))
        n = 0

        for res in sorted((ranges), reverse=True):
            n += 1
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_louvain == fixed_clus_count:
                break
            if n == len(ranges):
                print('Failed to find the suitable resolution !!! Please change the range !!!')

        return res



# Accuracy_calculate
def Accuracy_calculate(adata, mclust_label, ground_truth_label):

    label_encoder = LabelEncoder()
    mclust_labels = label_encoder.fit_transform(adata.obs[mclust_label].tolist())
    ground_truth_labels = label_encoder.fit_transform(adata.obs[ground_truth_label].tolist())
    nmi = normalized_mutual_info_score(mclust_labels, ground_truth_labels)
    ari = adjusted_rand_score(mclust_labels, ground_truth_labels)
    
    return nmi, ari


# metadata obtain
def DLPFC_metadata(metapath, adata):
    

    ann_df = pd.read_csv( metapath + '_truth.txt',
                            sep = '\t',
                            header = None,
                            index_col=0)
        
    ann_df.columns = ['ground_truth']
    
    adata.obs['ground_truth'] = ann_df.loc[adata.obs_names, 'ground_truth']   
    adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]


    return adata




# mask edge with ratio
def random_edge_mask(adj, mask_prob, num_nodes):

    from torch_geometric.utils import add_self_loops
    # The input of random mask must not have eye !!!
    if torch.any(torch.diag(adj) != 0) == True:
        adj.fill_diagonal_(0)
    
    adj = sp.coo_matrix(adj)
    adj_ori = torch.tensor(list(zip(list(adj.row), list(adj.col))))
    num_edge = len(adj_ori)
    index = np.arange(len(adj_ori))
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_prob)
    pre_index = index[0:-mask_num]
    mask_index = index[-mask_num:]
    edge_index_train = adj_ori[pre_index].t()     # 2 * (num_edge * mask_prob)
    edge_index_mask = adj_ori[mask_index]     # (num_edge * 1-mask_prob) * 2
    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=num_nodes)

    return edge_index_train, edge_index_mask




def LossFig(Loss_cpu):
    
    import matplotlib.pyplot as plt
    import torch
    # Loss_cpu = Loss.cpu().detach().numpy()
    plt.figure()
    plt.plot(torch.tensor(Loss_cpu),'b',label = 'loss')
    # plt.plot(Loss,'b',label = 'loss')
    plt.ylabel('loss')
    plt.xlabel('iter_num')
    plt.grid()
    plt.show()




# cluster number select
def cluster_select(adata, min_clust, max_clust, figsave):

    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import torch
    
    f = open(figsave + 'sil_resu.txt','w')
    f.write('n_clust'+'\t'+'silhouette'+'\n')

    cluster_sil = {}

    for num_categories in range(min_clust, max_clust+1):

        sc.pp.neighbors(adata, use_rep='hidden_re')
        sc.tl.umap(adata)
        adata = mclust_R(adata, used_obsm='hidden_re', num_cluster = num_categories)

        cluster_labels = adata.obs['mclust']
        silhouette_avg = silhouette_score(adata.obsm['hidden_re'], cluster_labels)
        cluster_sil[num_categories] = silhouette_avg

        f.write(str(num_categories)+'\t'+str(silhouette_avg) +'\n')

    f.close()

    return cluster_sil


