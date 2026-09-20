"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Muon applies momentum SGD to matrix-shaped hidden weights and orthogonalizes
each update with a Newton-Schulz iteration before applying it, which keeps the
update's singular values near 1 (an approximate steepest descent under the
spectral norm). Reference: Keller Jordan et al., 2024
(https://kellerjordan.github.io/posts/muon/).

Only 2D+ hidden weights benefit; embeddings, biases, and per-channel gains
(weight_norm's weight_g) keep AdamW. The Muon class below therefore carries
two param groups, one per update rule, so schedulers and checkpointing see a
single ordinary optimizer.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate UV^T of the SVD of G via a quintic Newton-Schulz iteration.

    Runs in float32 (not bfloat16 as in the reference) for correctness on
    pre-Ampere GPUs; the matrices here are small enough that this is cheap.
    The iteration's singular values land in roughly [0.7, 1.2], which is all
    Muon needs.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def split_matrix_params(model):
    """Partition parameters into (muon_params, adamw_params).

    Muon gets matrix-shaped hidden weights: ndim >= 2, not an nn.Embedding
    weight, not weight_norm's per-channel gain (weight_g). Everything else
    (biases, gains, embeddings) goes to AdamW.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    emb_ids = set()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            emb_ids.update(id(p) for p in m.parameters(recurse=False))
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and id(p) not in emb_ids and not name.endswith("weight_g"):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


class Muon(Optimizer):
    """Muon for matrix params, decoupled AdamW for everything else.

    Conv kernels (out, in, k) are flattened to (out, in*k) for the
    orthogonalization, following the Moonlight/Kimi treatment. The update is
    scaled by sqrt(max(1, rows/cols)) so its RMS matches AdamW's, letting one
    LR schedule drive both groups.
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_lr=1e-3,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        weight_decay=0.01,
    ):
        groups = [
            dict(
                params=list(muon_params),
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
                use_muon=True,
            ),
            dict(
                params=list(adamw_params),
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=weight_decay,
                use_muon=False,
            ),
        ]
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_group_step(group)
            else:
                self._adamw_group_step(group)
        return loss

    def _muon_group_step(self, group):
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(group["momentum"]).add_(g)
            update = (
                g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
            )
            u2 = update.reshape(update.size(0), -1)
            u2 = zeropower_via_newtonschulz5(u2, steps=group["ns_steps"])
            scale = max(1.0, u2.size(0) / u2.size(1)) ** 0.5
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
            p.add_(u2.reshape_as(p), alpha=-group["lr"] * scale)

    def _adamw_group_step(self, group):
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq / bc2).sqrt_().add_(group["eps"])
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
            p.addcdiv_(exp_avg, denom, value=-group["lr"] / bc1)
