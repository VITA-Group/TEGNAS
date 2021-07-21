import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .genotypes import Structure
from .search_cells import NAS201SearchCell as SearchCell
from ..cell_operations import ResNetBasicblock


def cal_entropy(logit: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    ~
    :param logit: An unnormalized vector.
    :param dim: ~
    :return: entropy
    """
    prob = F.softmax(logit, dim=dim)
    log_prob = F.log_softmax(logit, dim=dim)

    entropy = -(log_prob * prob).sum(-1, keepdim=False)

    return entropy


class TinyNetworkDarts(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats, depth=-1,
                 use_stem=True):
        super(TinyNetworkDarts, self).__init__()
        self._C = C
        self._layerN = N  # number of stacked cell at each stage
        self.max_nodes = max_nodes
        self.use_stem = use_stem
        self.stem = nn.Sequential(
            nn.Conv2d(min(3, C), C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            # nn.ReLU(inplace=True)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if depth > 0 and index >= depth: break
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(
                        num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_parameters = nn.Parameter(1e-3 * torch.randn(num_edge, len(search_space)))
        # self.conv_ending = nn.Sequential(
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(C_prev, C_prev, kernel_size=1, padding=0, bias=False)  # TODO just for receving & calculating the activation pattern from the last cell's conv
        #         )

    def entropy(self, mean=True):
        if mean:
            return cal_entropy(self.arch_parameters, -1).mean().view(-1)
        else:
            return cal_entropy(self.arch_parameters, -1)

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def no_grad(self, modules=['stem', 'cells', 'lastact', 'classifier']):
        if 'stem' in modules:
            for param in self.stem.parameters():
                param.requires_grad = False
        if 'cells' in modules:
            for param in self.cells.parameters():
                param.requires_grad = False
        if 'lastact' in modules:
            for param in self.lastact.parameters():
                param.requires_grad = False
        # for param in self.global_pooling.parameters():
        #     param.requires_grad = False
        if 'classifier' in modules:
            for param in self.classifier.parameters():
                param.requires_grad = False
        return

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def get_alphas(self):
        return [self.arch_parameters]

    def set_alphas(self, arch_parameters):
        self.arch_parameters.data.copy_(arch_parameters[0].data)

    def show_alphas(self):
        with torch.no_grad():
            return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self.arch_parameters, dim=-1).cpu())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__,
                                                                                        **self.__dict__))

    def genotype(self, get_random=False, hardwts=None):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    if hardwts is not None:
                        weights = hardwts[self.edge2index[node_str]]
                        op_name = self.op_names[weights.argmax().item()]
                    elif get_random:
                        op_name = random.choice(self.op_names)
                    else:
                        weights = self.arch_parameters[self.edge2index[node_str]]
                        op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def prune_arch_params(self):
        # lim = float('inf')
        lim = 10 ** 7
        lim_ = 10 ** 6
        alphas = self.arch_parameters.data.clone()
        alphas[alphas < -lim_] = lim
        for edge in range(len(alphas)):
            alpha = alphas[edge]
            assert (alpha < lim).sum() >= 1, "edge: %d" % edge
        if (alphas < lim).sum() == len(alphas):
            # all edges only remains one operator
            return
        edge, idx = torch.where(alphas == alphas.min())
        while (alphas[edge] < lim).sum() == 1:
            alphas[edge, idx] = lim
            # this edge only remains one operator
            edge, idx = torch.where(alphas == alphas.min())
        self.arch_parameters.data[edge, idx] = -lim
        # alphas[alphas == lim] = -lim

    def prune_arch_params_structural(self):
        # lim = float('inf')
        lim = 10 ** 7
        lim_ = 10 ** 6
        alphas = self.arch_parameters.data.clone()
        alphas[alphas < -lim_] = lim
        for edge in range(len(alphas)):
            alpha = alphas[edge]
            if (alpha < lim).sum() == 1: continue
            idx = alpha.argmin()
            self.arch_parameters.data[edge, idx] = -lim
        # alphas[alphas == lim] = -lim

    def _derive_hardwts(self):
        probs = nn.functional.softmax(self.arch_parameters, dim=1)
        one_h, index = self.derive_hardwts_no_grad()
        hardwts = one_h - probs.detach() + probs
        return hardwts, index

    def derive_hardwts_no_grad(self):
        probs = nn.functional.softmax(self.arch_parameters, dim=1)
        index = probs.max(-1, keepdim=True)[1]
        one_h = torch.zeros_like(probs).scatter_(-1, index, 1.0)
        return one_h, index

    def _random_hardwts(self):
        while True:
            gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
            logits = (self.arch_parameters.log_softmax(dim=1) + gumbels) / 100000
            probs = nn.functional.softmax(logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                continue
            else:
                break
        return hardwts, index

    def _forward_hardwts(self, inputs, hardwts, index):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_gdas(feature, hardwts, index)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

    def forward_prob_nas(self, inputs, index):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_prob_nas(feature, index)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

    def forward_shallow(self, inputs):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        feature = self.cell_shallow(inputs, alphas)
        # feature = self.stem(inputs)
        # for i, cell in enumerate(self.cells[:1]):
        #     if isinstance(cell, SearchCell):
        #         feature = cell(feature, alphas)
        #     else:
        #         feature = cell(feature)

    def forward(self, inputs, return_features=False):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        features_all = []
        if self.use_stem:
            feature = self.stem(inputs)
        else:
            feature = inputs
        # features_all.append(feature.detach())
        features_all.append(feature)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)
            # features_all.append(feature.detach())
            features_all.append(feature)

        # TODO receiving the last cell's activation pattern
        # _ = self.conv_ending(feature)

        out = self.lastact(feature)  # TODO change to post-activation
        out = self.global_pooling(out)
        # out = self.global_pooling( feature )
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        if return_features:
            return out, logits, features_all
        else:
            return out, logits

    def forward_random_path(self, inputs):
        hardwts, index = self._random_hardwts()
        out, logits = self._forward_hardwts(inputs, hardwts, index)
        return out, logits, hardwts


# Learning towards Minimum Hyperspherical Energy
# https://github.com/wy1iu/MHE/blob/master/code/architecture.py
# https://github.com/rmlin/CoMHE/blob/master/adversarial_projection/architecture.py
@torch.no_grad()
def hypersphereenergy(filt, filt_target=None, paired=False, model=None, power='0', device='cuda:0', mem_efficient=False):
    # TODO if filt_target is None: filt energy itself; if filt_target is not None: filt energy against filt_target
    # TODO paired: if size of filt and filt_target are same: only compare diagonal
    # tensorflow: [ksize, ksize, n_input, n_filt]
    # pytorch: [n_filt, n_input, ksize, ksize]
    n_filt = filt.size(0)
    filt = filt.view(n_filt, -1)
    if filt_target is not None:
        n_filt_target = filt_target.size(0)
        filt_target = filt_target.view(n_filt_target, -1)
        assert filt.size(1) == filt_target.size(1), str(filt.size()) + " v.s. " + str(filt_target.size())
    if paired:
        assert n_filt == n_filt_target
    # TODO half_mhe not ready
    # if model =='half_mhe':
    #     filt_neg = filt * -1
    #     filt = torch.cat([filt, filt_neg], dim=0)
    #     n_filt *= 2
    filt_norm = torch.sqrt(torch.sum(filt*filt, 1, keepdim=True) + 1e-4)
    if filt_target is not None:
        filt_norm_target = torch.sqrt(torch.sum(filt_target*filt_target, 1, keepdim=True) + 1e-4)
    if filt_target is None:
        norm_mat = torch.einsum('nc,mc->nm', [filt_norm, filt_norm])
        inner_pro = torch.einsum('nc,mc->nm', [filt, filt])
    else:
        norm_mat = torch.einsum('nc,mc->nm', [filt_norm, filt_norm_target])
        inner_pro = torch.einsum('nc,mc->nm', [filt, filt_target])
    if mem_efficient: del filt_norm; torch.cuda.empty_cache()
    inner_pro /= norm_mat
    if mem_efficient: del norm_mat; torch.cuda.empty_cache()
    if power =='0':
        cross_terms = torch.clamp(2.0 - 2.0 * inner_pro, 1e-4) # convert similarity to distance
        if mem_efficient: del inner_pro; torch.cuda.empty_cache()
        # final -= torch.tril(final)#, diagonal=-1)
        if filt_target is None:
            final = -torch.log(cross_terms)
            final = torch.tril(final, diagonal=-1)
            cnt = n_filt * (n_filt - 1) / 2.0
        else:
            final = -torch.log(cross_terms)
            cnt = n_filt * n_filt_target
        if paired:
            assert final.size(0) == final.size(1)
            final = torch.diagonal(final)
            cnt = n_filt
        loss = 1 * final.sum() / cnt
        if mem_efficient: del final; torch.cuda.empty_cache()
    elif power =='1':
        cross_terms = torch.clamp(2.0 - 2.0 * inner_pro, 1e-4) # + torch.diag(torch.ones(n_filt)).to(device))
        if mem_efficient: del inner_pro; torch.cuda.empty_cache()
        if filt_target is None:
            final = torch.pow(cross_terms + torch.diag(torch.ones(n_filt)).to(device), torch.ones_like(cross_terms) * (-0.5))
            final = torch.tril(final, diagonal=-1)
            cnt = n_filt * (n_filt - 1) / 2.0
        else:
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            cnt = n_filt * n_filt_target
        if paired:
            assert final.size(0) == final.size(1)
            final = torch.diagonal(final)
            cnt = n_filt
        # final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
        # final -= torch.tril(final)
        # cnt = n_filt * (n_filt - 1) / 2.0
        loss = 1 * final.sum() / cnt
        if mem_efficient: del final; torch.cuda.empty_cache()
    elif power =='2':
        # cross_terms = (torch.clamp(2.0 - 2.0 * inner_pro, 1e-4) + torch.diag(torch.ones(n_filt)).to(device))
        # final = torch.pow(cross_terms, torch.ones_like(cross_terms).to(device) * (-1))
        # final -= torch.tril(final)
        # cnt = n_filt * (n_filt - 1) / 2.0
        # loss = 1 * final.sum() / cnt
        cross_terms = torch.clamp(2.0 - 2.0 * inner_pro, 1e-4) # + torch.diag(torch.ones(n_filt)).to(device))
        if mem_efficient: del inner_pro; torch.cuda.empty_cache()
        if filt_target is None:
            final = torch.pow(cross_terms + torch.diag(torch.ones(n_filt)).to(device), torch.ones_like(cross_terms) * (-1))
            final = torch.tril(final, diagonal=-1)
            cnt = n_filt * (n_filt - 1) / 2.0
        else:
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-1))
            cnt = n_filt * n_filt_target
        if paired:
            assert final.size(0) == final.size(1)
            final = torch.diagonal(final)
            cnt = n_filt
        loss = 1 * final.sum() / cnt
        if mem_efficient: del final; torch.cuda.empty_cache()
    elif power =='a0':
        acos = torch.acos(inner_pro)/math.pi
        acos += 1e-4
        if mem_efficient: del inner_pro; torch.cuda.empty_cache()
        final = -torch.log(acos)
        if filt_target is None:
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
        elif paired:
            assert final.size(0) == final.size(1)
            final = torch.diagonal(final)
            cnt = n_filt
        else:
            cnt = n_filt * n_filt_target
        loss = 1 * final.sum() / cnt
        if mem_efficient: del final; torch.cuda.empty_cache()
    elif power =='a1':
        acos = torch.acos(inner_pro)/math.pi
        acos += 1e-4
        if mem_efficient: del inner_pro; torch.cuda.empty_cache()
        final = torch.pow(acos, torch.ones_like(acos) * (-1))
        if filt_target is None:
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
        elif paired:
            assert final.size(0) == final.size(1)
            final = torch.diagonal(final)
            cnt = n_filt
        else:
            cnt = n_filt * n_filt_target
        # final -= torch.tril(final)
        # cnt = n_filt * (n_filt - 1) / 2.0
        loss = 1e-1 * final.sum() / cnt
        if mem_efficient: del final; torch.cuda.empty_cache()
    elif power =='a2':
        acos = torch.acos(inner_pro)/math.pi
        acos += 1e-4
        if mem_efficient: del inner_pro; torch.cuda.empty_cache()
        final = torch.pow(acos, torch.ones_like(acos) * (-2))
        if filt_target is None:
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
        elif paired:
            assert final.size(0) == final.size(1)
            final = torch.diagonal(final)
            cnt = n_filt
        else:
            cnt = n_filt * n_filt_target
        # final -= torch.tril(final)
        # cnt = n_filt * (n_filt - 1) / 2.0
        loss = 1e-1 * final.sum() / cnt
        if mem_efficient: del final; torch.cuda.empty_cache()
    return loss