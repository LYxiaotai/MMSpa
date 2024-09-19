import torch
import torch.backends
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn





class En_DecoderGAT(nn.Module):
    def __init__(self, 
                in_channels, 
                hidden_channels, 
                out_channels,
                num_layers, dropout = 0.0):
        super(En_DecoderGAT, self).__init__()

        self.gat_layers = torch.nn.ModuleList()

        self.gat_layers.append(L_GraphAttentionLayer(in_channels, hidden_channels, dropout=0.0))
        for _ in range(1, num_layers - 1):
            self.gat_layers.append(
                L_GraphAttentionLayer(hidden_channels, hidden_channels, dropout=0.0))
        self.gat_layers.append(L_GraphAttentionLayer(hidden_channels, out_channels, dropout=0.0))

        self.head = nn.Identity()
        self.dropout = dropout

    def forward(self, adj, x):
        h = x
        hidden_list = []
        for conv in self.gat_layers[:-1]:
            h = conv(h, adj)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hidden_list.append(h)
        h = self.gat_layers[-1](h, adj)
        hidden_list.append(F.elu(h))        

        return self.head(h), hidden_list
    






class L_GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, 
                 in_features, out_features, 
                 dropout):
        
        super(L_GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # self.lin
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))   # self.att
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.attentions = None

    def forward(self, h, adj, 
                attention = True,
                tried_attention = None):
        
        # print(h.shape)
        # print(self.W.shape)

        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        
        if attention == False:
            return Wh
        
        if tried_attention == None:
            e = self._prepare_attentional_mechanism_input(Wh)
            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            self.attentions = attention
            h_prime = torch.matmul(attention, Wh)
            return h_prime
        else:
            attention = tried_attention
            h_prime = torch.matmul(attention, Wh)
            return h_prime
        

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return torch.sigmoid(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'





