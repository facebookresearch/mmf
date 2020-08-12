# Copyright (c) Facebook, Inc. and its affiliates.

"""
The fusions module contains various Fusion techniques, some based on BLOCK:
Bilinear Superdiagonal Fusion for VQA and VRD. For e.g. LinearSum, ConcatMLP
etc taken from https://github.com/Cadene/block.bootstrap.pytorch#fusions.

For implementing your own fusion technique, you need to follow these steps:

.. code::
    from torch import nn
    from mmf.common.registry import registry
    from mmf.modules.fusions import Block
    from mmf.modules.fusions import LinearSum
    from mmf.modules.fusions import ConcatMLP
    from mmf.modules.fusions import MLB
    from mmf.modules.fusions import Mutan
    from mmf.modules.fusions import Tucker
    from mmf.modules.fusions import BlockTucker
    from mmf.modules.fusions import MFH
    from mmf.modules.fusions import MFB
    from mmf.modules.fusions import MCB

    @regitery.register_fusion("custom")
    class CustomFusion(nn.Module):
        def __init__(self, params=None):
            super().__init__("Custom")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.utils.general import get_chunks, get_sizes_list


class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool=True):
        super().__init__()
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        self.sketch1 = nn.Parameter(
            self.generate_sketch_matrix(
                torch.randint(output_dim, size=(input_dim1,)),
                2 * torch.randint(2, size=(input_dim1,)) - 1,
                input_dim1,
                output_dim,
            ),
            requires_grad=False,
        )
        self.sketch2 = nn.Parameter(
            self.generate_sketch_matrix(
                torch.randint(output_dim, size=(input_dim2,)),
                2 * torch.randint(2, size=(input_dim2,)) - 1,
                input_dim2,
                output_dim,
            ),
            requires_grad=False,
        )

    def generate_sketch_matrix(self, rand_h, rand_s, input_dim, output_dim):
        return torch.sparse.FloatTensor(
            torch.stack(
                [torch.arange(input_dim, out=torch.LongTensor()), rand_h.long()]
            ),
            rand_s.float(),
            [input_dim, output_dim],
        ).to_dense()

    def forward(self, x1, x2):
        assert len(x1.shape) == len(x2.shape)
        if len(x1.shape) == 4 and len(x2.shape) == 4:
            fft1 = torch.rfft(
                x1.permute(0, 2, 3, 1).matmul(self.sketch1), signal_ndim=1
            )
            fft2 = torch.rfft(
                x2.permute(0, 2, 3, 1).matmul(self.sketch2), signal_ndim=1
            )
        else:
            fft1 = torch.rfft(x1.matmul(self.sketch1), signal_ndim=1)
            fft2 = torch.rfft(x2.matmul(self.sketch2), signal_ndim=1)
        fft_product = torch.stack(
            [
                fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1],
                fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0],
            ],
            dim=-1,
        )
        cbp = (
            torch.irfft(fft_product, signal_ndim=1, signal_sizes=(self.output_dim,))
            * self.output_dim
        )
        if len(x1.shape) == 4 and len(x2.shape) == 4:
            cbp = cbp.sum(dim=[1, 2]) if self.sum_pool else cbp.permute(0, 3, 1, 2)
        return cbp


class MLP(nn.Module):
    def __init__(self, input_dim, dimensions, activation="relu", dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = F.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x


@registry.register_fusion("block")
class Block(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=1600,
        chunks=20,
        rank=15,
        shared=False,
        dropout_input=0.0,
        dropout_pre_lin=0.0,
        dropout_output=0.0,
        pos_norm="before_cat",
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ["before_cat", "after_cat"]
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size * rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size * rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, m0, m1 in zip(
            range(len(self.sizes_list)), self.merge_linears0, self.merge_linears1
        ):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            m = m0(x0_c) * m1(x1_c)  # bsize x split_size*rank
            m = m.view(bsize, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == "before_cat":
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z, p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == "after_cat":
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("block_tucker")
class BlockTucker(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=1600,
        chunks=20,
        shared=False,
        dropout_input=0.0,
        dropout_pre_lin=0.0,
        dropout_output=0.0,
        pos_norm="before_cat",
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ["before_cat", "after_cat"]
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if self.shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)

        self.sizes_list = get_sizes_list(mm_dim, chunks)
        bilinears = []
        for size in self.sizes_list:
            bilinears.append(nn.Bilinear(size, size, size))
        self.bilinears = nn.ModuleList(bilinears)
        self.linear_out = nn.Linear(self.mm_dim, self.output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.dropout_input:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, bilinear in enumerate(self.bilinears):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            z = bilinear(x0_c, x1_c)
            if self.pos_norm == "before_cat":
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z, p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == "after_cat":
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("mutan")
class Mutan(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=1600,
        rank=15,
        shared=False,
        normalize=False,
        dropout_input=0.0,
        dropout_pre_lin=0.0,
        dropout_output=0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.rank = rank
        self.output_dim = output_dim
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.normalize = normalize
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.merge_linear0 = nn.Linear(mm_dim, mm_dim * rank)
        if self.shared:
            self.linear1 = self.linear0
            self.merge_linear1 = self.merge_linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
            self.merge_linear1 = nn.Linear(mm_dim, mm_dim * rank)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        m0 = self.merge_linear0(x0)
        m1 = self.merge_linear1(x1)
        m = m0 * m1
        m = m.view(-1, self.rank, self.mm_dim)
        z = torch.sum(m, 1)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("tucker")
class Tucker(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=1600,
        shared=False,
        normalize=False,
        dropout_input=0.0,
        dropout_pre_lin=0.0,
        dropout_output=0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.bilinear = nn.Bilinear(mm_dim, mm_dim, mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = self.bilinear(x0, x1)

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("mlb")
class MLB(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=1200,
        activ_input="relu",
        activ_output="relu",
        normalize=False,
        dropout_input=0.0,
        dropout_pre_lin=0.0,
        dropout_output=0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 * x1

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("mfb")
class MFB(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=1200,
        factor=2,
        activ_input="relu",
        activ_output="relu",
        normalize=False,
        dropout_input=0.0,
        dropout_pre_norm=0.0,
        dropout_output=0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.factor = factor
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 * x1

        if self.dropout_pre_norm > 0:
            z = F.dropout(z, p=self.dropout_pre_norm, training=self.training)

        z = z.view(z.size(0), self.mm_dim, self.factor)
        z = z.sum(2)

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("mfh")
class MFH(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=1200,
        factor=2,
        activ_input="relu",
        activ_output="relu",
        normalize=False,
        dropout_input=0.0,
        dropout_pre_lin=0.0,
        dropout_output=0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0_0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1_0 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear0_1 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1_1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim * 2, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0_0(x[0])
        x1 = self.linear1_0(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_0_skip = x0 * x1

        if self.dropout_pre_lin:
            z_0_skip = F.dropout(
                z_0_skip, p=self.dropout_pre_lin, training=self.training
            )

        z_0 = z_0_skip.view(z_0_skip.size(0), self.mm_dim, self.factor)
        z_0 = z_0.sum(2)

        if self.normalize:
            z_0 = torch.sqrt(F.relu(z_0)) - torch.sqrt(F.relu(-z_0))
            z_0 = F.normalize(z_0, p=2)

        #
        x0 = self.linear0_1(x[0])
        x1 = self.linear1_1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_1 = x0 * x1 * z_0_skip

        if self.dropout_pre_lin > 0:
            z_1 = F.dropout(z_1, p=self.dropout_pre_lin, training=self.training)

        z_1 = z_1.view(z_1.size(0), self.mm_dim, self.factor)
        z_1 = z_1.sum(2)

        if self.normalize:
            z_1 = torch.sqrt(F.relu(z_1)) - torch.sqrt(F.relu(-z_1))
            z_1 = F.normalize(z_1, p=2)

        #
        cat_dim = z_0.dim() - 1
        z = torch.cat([z_0, z_1], cat_dim)
        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("mcb")
class MCB(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=16000,
        activ_output="relu",
        dropout_output=0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_output = activ_output
        self.dropout_output = dropout_output
        # Modules
        self.mcb = CompactBilinearPooling(input_dims[0], input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        z = self.mcb(x[0], x[1])
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("linear_sum")
class LinearSum(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        mm_dim=1200,
        activ_input="relu",
        activ_output="relu",
        normalize=False,
        dropout_input=0.0,
        dropout_pre_lin=0.0,
        dropout_output=0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 + x1

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion("concat_mlp")
class ConcatMLP(nn.Module):
    def __init__(
        self, input_dims, output_dim, dimensions=None, activation="relu", dropout=0.0
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.input_dim = sum(input_dims)
        if dimensions is None:
            dimensions = [500, 500]
        self.dimensions = dimensions + [output_dim]
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.mlp = MLP(self.input_dim, self.dimensions, self.activation, self.dropout)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if x[0].dim() == 3 and x[1].dim() == 2:
            x[1] = x[1].unsqueeze(1).reshape_as(x[0])
        if x[1].dim() == 3 and x[0].dim() == 2:
            x[0] = x[0].unsqueeze(1).reshape_as(x[1])
        z = torch.cat(x, dim=x[0].dim() - 1)
        z = self.mlp(z)
        return z
