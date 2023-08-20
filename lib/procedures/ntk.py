import numpy as np
import torch


def get_ntk_n(loader, networks, loader_val=None, train_mode=False, num_batch=-1, num_classes=100):
    device = torch.cuda.current_device()
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads_x = [[] for _ in range(len(networks))]
    cellgrads_x = [[] for _ in range(len(networks))]; cellgrads_y = [[] for _ in range(len(networks))]
    ntk_cell_x = []; ntk_cell_yx = []; prediction_mses = []
    targets_x_onehot_mean = []; targets_y_onehot_mean = []
    for i, (inputs, targets) in enumerate(loader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        targets = targets.cuda(device=device, non_blocking=True)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
        targets_x_onehot_mean.append(targets_onehot_mean)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                cellgrad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                        if "cell" in name:
                            cellgrad.append(W.grad.view(-1).detach())
                grads_x[net_idx].append(torch.cat(grad, -1))
                cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).cuda()
                if len(cellgrads_x[net_idx]) == 0:
                    cellgrads_x[net_idx] = [cellgrad]
                else:
                    cellgrads_x[net_idx].append(cellgrad)
                network.zero_grad()
                torch.cuda.empty_cache()
    targets_x_onehot_mean = torch.cat(targets_x_onehot_mean, 0)
    # cell's NTK #####
    for _i, grads in enumerate(cellgrads_x):
        grads = torch.stack(grads, 0)
        _ntk = torch.einsum('nc,mc->nm', [grads, grads])
        ntk_cell_x.append(_ntk)
        cellgrads_x[_i] = grads
    # NTK cond
    grads_x = [torch.stack(_grads, 0) for _grads in grads_x]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads_x]
    conds_x = []
    for ntk in ntks:
        try:
            eigenvalues, _ = torch.symeig(ntk)  # ascending
            _cond = eigenvalues[-1] / eigenvalues[0]
            if torch.isnan(_cond):
                conds_x.append(-1) # bad gradients
            else:
                conds_x.append(_cond.item())
        except RuntimeError:
            conds_x.append(-1) # bad gradients
    # Val / Test set
    if loader_val is not None:
        for i, (inputs, targets) in enumerate(loader_val):
            if num_batch > 0 and i >= num_batch: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            targets = targets.cuda(device=device, non_blocking=True)
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
            targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
            targets_y_onehot_mean.append(targets_onehot_mean)
            for net_idx, network in enumerate(networks):
                network.zero_grad()
                inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
                logit = network(inputs_)
                if isinstance(logit, tuple):
                    logit = logit[1]  # 201 networks: return features and logits
                for _idx in range(len(inputs_)):
                    logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                    cellgrad = []
                    for name, W in network.named_parameters():
                        if 'weight' in name and W.grad is not None and "cell" in name:
                            cellgrad.append(W.grad.view(-1).detach())
                    cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).cuda()
                    if len(cellgrads_y[net_idx]) == 0:
                        cellgrads_y[net_idx] = [cellgrad]
                    else:
                        cellgrads_y[net_idx].append(cellgrad)
                    network.zero_grad()
                    torch.cuda.empty_cache()
        targets_y_onehot_mean = torch.cat(targets_y_onehot_mean, 0)
        for _i, grads in enumerate(cellgrads_y):
            grads = torch.stack(grads, 0)
            cellgrads_y[_i] = grads
        for net_idx in range(len(networks)):
            if cellgrads_y[net_idx].sum() == 0 or cellgrads_x[net_idx].sum() == 0:
                # bad gradients
                prediction_mses.append(-1)
                continue
            try:
                _ntk_yx = torch.einsum('nc,mc->nm', [cellgrads_y[net_idx], cellgrads_x[net_idx]])
                PY = torch.einsum('jk,kl,lm->jm', _ntk_yx, torch.inverse(ntk_cell_x[net_idx]), targets_x_onehot_mean)
                prediction_mses.append(((PY - targets_y_onehot_mean)**2).sum(1).mean(0).item())
            except RuntimeError:
                # RuntimeError: inverse_gpu: U(1,1) is zero, singular U.
                # prediction_mses.append(((targets_y_onehot_mean)**2).sum(1).mean(0).item())
                prediction_mses.append(-1) # bad gradients
    ######
    if loader_val is None:
        return conds_x
    else:
        return conds_x, prediction_mses
