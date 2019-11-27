import typing as t

import torch
from torch import nn

from .layers import *


class GraphInf(nn.Module):
    def __init__(
        self,
        num_in_feat: int,  #
        num_c_feat: int,
        num_embeddings: int,  #
        casual_hidden_sizes: t.Iterable,  #
        num_botnec_feat: int,  # 16 x 4
        num_k_feat: int,  # 16
        num_dense_layers: int,
        num_out_feat: int,
        num_z_feat: int,
        activation: str='elu',
        use_cuda: bool=True
    ):
        super().__init__()
        self.num_in_feat = num_in_feat
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_in_feat, num_embeddings)
        self.c_embedding = nn.Embedding(num_c_feat, num_embeddings)
        self.casual_hidden_sizes = casual_hidden_sizes
        self.num_botnec_feat = num_botnec_feat
        self.num_k_feat = num_k_feat
        self.num_dense_layers = num_dense_layers
        self.num_out_feat = num_out_feat
        self.activation = activation
        self.num_z_feat = num_z_feat
        self.use_cuda = use_cuda

        self.encode_dense = DenseNet(  # for encoding
            num_feat=self.num_embeddings,
            casual_hidden_sizes=self.casual_hidden_sizes,
            num_botnec_feat=self.num_botnec_feat,
            num_k_feat=self.num_k_feat,
            num_dense_layers=self.num_dense_layers,
            num_out_feat=self.num_out_feat,
            activation=self.activation
        )

        self.decode_dense = DenseNet(  # for decoding
            num_feat=self.num_z_feat + self.num_embeddings,
            casual_hidden_sizes=self.casual_hidden_sizes,
            num_botnec_feat=self.num_botnec_feat,
            num_k_feat=self.num_k_feat,
            num_dense_layers=self.num_dense_layers,
            num_out_feat=self.num_in_feat,
            activation=self.activation
        )

        self.c_dense = DenseNet(
            num_feat=self.num_embeddings,
            casual_hidden_sizes=self.casual_hidden_sizes,
            num_botnec_feat=self.num_botnec_feat,
            num_k_feat=self.num_k_feat,
            num_dense_layers=self.num_dense_layers,
            num_out_feat=self.num_out_feat
        )

        self.softplus = nn.Softplus()

        self.fc1 = nn.Linear(
            self.num_out_feat,
            self.num_z_feat
        )

        self.fc2 = nn.Linear(
            self.num_out_feat,
            self.num_z_feat
        )

        self.cfc1 = nn.Linear(
            self.num_out_feat,
            self.num_z_feat
        )

        self.cfc2 = nn.Linear(
            self.num_out_feat,
            self.num_z_feat
        )

    def encode(self, feat, adj):
        h1 = self.encode_dense(feat, adj)
        return self.fc1(h1), self.fc2(h1)

    def c_encode(self, feat_c, adj):
        h2 = self.c_dense(feat_c, adj)
        return self.cfc1(h2), self.cfc2(h2)

    def decode(self, z, adj):
        x_recon = self.decode_dense(z, adj)
        return x_recon

    def reparametrize(
        self,
        mu,
        var
    ):
        std = var.sqrt()
        if self.use_cuda and torch.cuda.is_available():
            eps = torch.FloatTensor(std.size()).normal_()
            eps = eps.to(std.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(
        self,
        feat_o: torch.Tensor,
        feat_c: torch.Tensor,
        adj: torch.sparse.FloatTensor,
    ) -> torch.Tensor:
        feat_o = self.embedding(feat_o)
        feat_c = self.c_embedding(feat_c)
        mu2, x_2 = self.c_encode(feat_c, adj)
        mu1, x_1 = self.encode(feat_o, adj)
        var1, var2 = self.softplus(x_1), self.softplus(x_2)
        z = self.reparametrize(mu1, var1)
        z_c = torch.cat([z, feat_c], dim=-1)
        x_recon = self.decode(z_c, adj)
        return x_recon, mu1, var1, mu2, var2

    def inf(
        self,
        feat_c: torch.Tensor,
        adj: torch.sparse.FloatTensor,
    ) -> torch.Tensor:
        feat_c = self.c_embedding(feat_c)
        mu2, x_2 = self.c_encode(feat_c, adj)
        var2 = self.softplus(x_2)
        z = self.reparametrize(mu2, var2)
        z_c = torch.cat([z, feat_c], dim=-1)
        x_recon = self.decode(z_c, adj)
        return x_recon, mu2, var2

    def reconstrcut(
        self,
        feat_o: torch.Tensor,
        feat_c: torch.Tensor,
        adj: torch.sparse.FloatTensor
    ) -> torch.Tensor:
        feat_o = self.embedding(feat_o)
        feat_c = self.c_embedding(feat_c)
        mu1, x_1 = self.encode(feat_o, adj)
        var1 = self.softplus(x_1)
        z = self.reparametrize(mu1, var1)
        z_c = torch.cat([z, feat_c], dim=-1)
        x_recon = self.decode(z_c, adj)
        return x_recon, mu1, var1
