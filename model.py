import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x
        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)
        return x2 + shortcut


class InteractionMask(nn.Module):
    """
    Generates sparse interaction masks via asymmetric convolutions.
    Paper Section 3.2.2: threshold ξ=0.5, values < threshold set to 0.
    
    NOTE: Original repo uses soft masking (keeps sigmoid values above threshold).
    This differs from paper Eq. 11-12 which describes binary masks.
    We follow the original repo implementation here.
    """

    def __init__(self, number_asymmetric_conv_layer=2, spatial_channels=4, temporal_channels=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(spatial_channels, spatial_channels)
            )
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):

        assert len(dense_temporal_interaction.shape) == 4   # (T, num_heads, N, N)
        assert len(dense_spatial_interaction.shape) == 4    # (N, num_heads, T, T)

        for j in range(self.number_asymmetric_conv_layer):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask)
        temporal_zero = torch.zeros_like(temporal_interaction_mask)

        spatial_interaction_mask = torch.where(
            spatial_interaction_mask > threshold,
            spatial_interaction_mask,
            spatial_zero
        )
        temporal_interaction_mask = torch.where(
            temporal_interaction_mask > threshold,
            temporal_interaction_mask,
            temporal_zero
        )

        return spatial_interaction_mask, temporal_interaction_mask


class ZeroSoftmax(nn.Module):
    """
    Zero-preserving softmax (Paper Eq. 14).
    """

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class SelfAttention(nn.Module):
    """
    Multi-head self-attention for computing dense interaction scores.
    Paper Section 3.2.2: Eq. 2-5 (spatial), Eq. 6-8 (temporal).
    """

    def __init__(self, in_dims=4, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.register_buffer('scaled_factor', torch.sqrt(torch.tensor(float(d_model))))
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads
        self.d_model = d_model

    def split_heads(self, x):
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=False, multi_head=False):
        assert len(x.shape) == 3

        embeddings = self.embedding(x)
        query = self.query(embeddings)
        key = self.key(embeddings)

        if multi_head:
            query = self.split_heads(query)
            key = self.split_heads(key)
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))

        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:
            mask_mat = torch.ones_like(attention)
            attention = attention * torch.tril(mask_mat)

        return attention, embeddings


class SpatialTemporalFusion(nn.Module):
    """
    Fuses spatial attention scores across time dimension.
    Paper Section 3.2.2.
    """

    def __init__(self, obs_len=10):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        return x.squeeze()


class SparseWeightedAdjacency(nn.Module):
    """
    Generates normalized sparse spatial and temporal adjacency matrices.
    Paper Section 3.2.2: Eq. 2-15.

    Input graph shape: (T, N, 5) = [pos_enc, LON_rel, LAT_rel, SOG_rel, Heading_rel]
    - spatial_graph uses graph[:, :, 1:] → (T, N, 4) velocity features
    - temporal_graph uses full graph     → (N, T, 5)
    """

    def __init__(self, spa_in_dims=4, tem_in_dims=5, embedding_dims=64, obs_len=10,
                 dropout=0, number_asymmetric_conv_layer=2, num_heads=4):
        super(SparseWeightedAdjacency, self).__init__()

        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims, num_heads=num_heads)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims, num_heads=num_heads)

        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)

        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            spatial_channels=num_heads,
            temporal_channels=num_heads
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()

    def forward(self, graph, identity):
        assert len(graph.shape) == 3

        spatial_graph  = graph[:, :, 1:]          # (T, N, 4)
        temporal_graph = graph.permute(1, 0, 2)   # (N, T, 5)

        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(
            spatial_graph, multi_head=True
        )
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(
            temporal_graph, multi_head=True
        )

        st_interaction = self.spa_fusion(
            dense_spatial_interaction.permute(1, 0, 2, 3)
        ).permute(1, 0, 2, 3)

        ts_interaction = dense_temporal_interaction

        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)

        spatial_mask  = spatial_mask  + identity[0].unsqueeze(1)
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)

        normalized_spatial_adjacency_matrix = self.zero_softmax(
            dense_spatial_interaction * spatial_mask, dim=-1
        )
        normalized_temporal_adjacency_matrix = self.zero_softmax(
            dense_temporal_interaction * temporal_mask, dim=-1
        )

        return (normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix,
                spatial_embeddings, temporal_embeddings)


class GraphConvolution(nn.Module):
    """
    Single graph convolution layer: H' = σ(A · H · W)
    Paper Section 3.2.3.
    """

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout

    def forward(self, graph, adjacency):
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout, training=self.training)
        return gcn_features


class SparseGraphConvolution(nn.Module):
    """
    Two-path sparse GCN with simple addition fusion.
    Paper Section 3.2.3, Eq. 16-19.
    """

    def __init__(self, in_dims=4, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):
        graph = graph[:, :, :, 1:]              # [1, T, N, 4]

        spa_graph = graph.permute(1, 0, 2, 3)      # [T, 1, N, 4]
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # [N, 1, T, 4]

        gcn_spatial_layer1 = self.spatial_temporal_sparse_gcn[0](
            spa_graph, normalized_spatial_adjacency_matrix
        )  # [T, num_heads, N, emb]

        gcn_spatial_layer1_perm = gcn_spatial_layer1.permute(2, 1, 0, 3)  # [N, num_heads, T, emb]

        gcn_spatial_temporal_features = self.spatial_temporal_sparse_gcn[1](
            gcn_spatial_layer1_perm, normalized_temporal_adjacency_matrix
        )

        gcn_temporal_layer1 = self.temporal_spatial_sparse_gcn[0](
            tem_graph, normalized_temporal_adjacency_matrix
        )  # [N, num_heads, T, emb]

        gcn_temporal_layer1_perm = gcn_temporal_layer1.permute(2, 1, 0, 3)  # [T, num_heads, N, emb]

        gcn_temporal_spatial_features = self.temporal_spatial_sparse_gcn[1](
            gcn_temporal_layer1_perm, normalized_spatial_adjacency_matrix
        )

        x = gcn_spatial_layer1 + gcn_temporal_layer1_perm  # [T, num_heads, N, emb]

        H = x.permute(2, 0, 1, 3)  # [N, T, num_heads, emb]

        return H


class TCN(nn.Module):
    """
    Temporal Convolutional Network using Conv2d.
    Paper Section 3.2.4: causal left padding.

    FIX: only the first layer maps fin→fout.
         Subsequent layers map fout→fout.
         This prevents channel mismatch when fin != fout (e.g. obs_len=8, pred_len=12).
    """

    def __init__(self, fin, fout, layers=3, ksize=3):
        super(TCN, self).__init__()
        self.fin   = fin
        self.fout  = fout
        self.layers = layers
        self.ksize  = ksize

        self.convs = nn.ModuleList()
        for i in range(self.layers):
            in_ch  = fin if i == 0 else fout   # first layer: fin→fout, rest: fout→fout
            out_ch = fout
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=self.ksize))

    def forward(self, x):
        # x: [N, fin, num_heads, emb]
        for conv in self.convs:
            x = F.pad(x, (self.ksize - 1, 0, self.ksize - 1, 0))
            x = conv(x)
        return x


class Encoder(nn.Module):
    """
    Gated TCN encoder (Paper Eq. 20):
    H(l+1) = residual + sigmoid(Wf * H) ⊗ tanh(Wg * H)

    FIX: residual projection added for when fin != fout,
         so the residual connection works correctly.
    """

    def __init__(self, fin, fout, layers=3, ksize=3):
        super(Encoder, self).__init__()
        self.tcnf = TCN(fin, fout, layers, ksize)
        self.tcng = TCN(fin, fout, layers, ksize)

        # Residual projection when fin != fout (e.g. obs_len=8 → pred_len=12)
        if fin != fout:
            self.residual_proj = nn.Conv2d(fin, fout, kernel_size=1)
        else:
            self.residual_proj = None

    def forward(self, x):
        # x: [N, fin, num_heads, emb]
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        f = torch.sigmoid(self.tcnf(x))
        g = torch.tanh(self.tcng(x))
        return residual + f * g


class TrajectoryModel(nn.Module):

    def __init__(self,
                 number_asymmetric_conv_layer=2, embedding_dims=64, number_gcn_layers=1,
                 dropout=0, obs_len=10, pred_len=10, out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.obs_len       = obs_len
        self.pred_len      = pred_len
        self.dropout       = dropout
        self.num_heads     = num_heads
        self.embedding_dims = embedding_dims

        gcn_hidden = embedding_dims // num_heads  # 16

        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            spa_in_dims=4,
            tem_in_dims=5,
            embedding_dims=embedding_dims,
            obs_len=obs_len,
            dropout=dropout,
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,
            num_heads=num_heads
        )

        self.stsgcn = SparseGraphConvolution(
            in_dims=4,
            embedding_dims=gcn_hidden,
            dropout=dropout
        )

        # fin=obs_len, fout=pred_len — TCN and Encoder now handle fin!=fout correctly
        self.encoder = Encoder(fin=obs_len, fout=pred_len)

        self.output = nn.Linear(embedding_dims, out_dims)

    def forward(self, graph, identity):
        """
        Args:
            graph:    [1, obs_len, N, 5]
            identity: [spatial (T,N,N), temporal (N,T,T)]
        """
        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, \
            spatial_embeddings, temporal_embeddings = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(0), identity)

        H = self.stsgcn(
            graph,
            normalized_spatial_adjacency_matrix,
            normalized_temporal_adjacency_matrix
        )  # [N, T, num_heads, gcn_hidden]

        features = self.encoder(H)  # [N, pred_len, num_heads, gcn_hidden]

        b, l, _, _ = features.shape
        features = features.contiguous().view(b, self.pred_len, -1)  # [N, pred_len, embedding_dims]

        prediction = self.output(features)  # [N, pred_len, out_dims=5]

        return prediction.permute(1, 0, 2).contiguous()  # [pred_len, N, 5]