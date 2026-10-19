import importlib.util
import os

import pytest
import torch
import yaml

from msfiddle.model_tcn import MS2FNet_tcn
from msfiddle.optim import Muon, split_matrix_params
from msfiddle.utils import ATOMS_INDEX_re, ATOMS_WEIGHT

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_trainer():
    path = os.path.join(REPO, "scripts", "train_tcn_gpus_cl.py")
    spec = importlib.util.spec_from_file_location("train_tcn_gpus_cl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


trainer = _load_trainer()


def _small_config():
    with open(os.path.join(REPO, "msfiddle/config/fiddle_tcn_qtof.yml")) as f:
        config = yaml.safe_load(f)["model"]
    config["tcn_channels"] = [
        4,
        4,
        8,
        16,
        32,
        64,
    ]  # keeps sum == 2 x last, as the fc input assumes
    config["tcn_dropout"] = 0.0  # deterministic forward for exact comparison
    return config


def _batches(n, seed=0):
    """n paired samples in the CL batch layout, as one list of per-sample tensors."""
    g = torch.Generator().manual_seed(seed)
    L = 7520
    n_atoms = len(ATOMS_INDEX_re)

    def side():
        y = torch.randint(0, 6, (n, n_atoms), generator=g).float()
        y[:, 0] += 1  # at least one C so the H/C target is finite
        x = torch.rand(n, L, 1, generator=g)
        mass = torch.rand(n, generator=g) * 500 + 100
        env = torch.zeros(n, 3)
        env[:, 0] = mass
        env[:, 1] = torch.rand(n, generator=g) * 50
        env[:, 2] = torch.randint(1, 7, (n,), generator=g).float()
        return y, x, mass, env

    y, x, mass, env = side()
    y2, x2, mass2, env2 = side()
    label = torch.randint(0, 2, (n,), generator=g).float()
    ids = [f"s{i}" for i in range(n)]
    return ids, y, x, mass, env, ids, y2, x2, mass2, env2, label


def _slice(full, a, b):
    return tuple(t[a:b] for t in full)


class _RecordingOptimizer(torch.optim.SGD):
    """Records the (post-clip) gradients it is asked to apply; never updates.

    Accumulation is judged on what reaches the optimizer: Muon normalizes each
    matrix's update, so for matrices whose true gradient is ~0 it turns float
    rounding noise into a unit-size step, and post-step parameters are not a
    meaningful comparison.
    """

    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.recorded = []

    @torch.no_grad()
    def step(self, closure=None):
        self.recorded.append(
            torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for g in self.param_groups
                    for p in g["params"]
                ]
            ).clone()
        )
        # Never update: the warmup scheduler overrides lr, and every step must
        # be evaluated at the same weights for the comparisons to hold.
        return None


def _run(model, loader, grad_accum_steps, warmup_steps=100):
    opt = _RecordingOptimizer(model.parameters())
    sched = trainer.WarmUpScheduler(opt, warmup_steps, 1e-6)
    weight = torch.tensor(
        [ATOMS_WEIGHT[ATOMS_INDEX_re[i]] for i in range(len(ATOMS_INDEX_re))]
    )
    trainer.train_step(
        model,
        loader,
        opt,
        sched,
        warmup_steps,
        torch.device("cpu"),
        batch_size=None,
        weight=weight,
        grad_accum_steps=grad_accum_steps,
    )
    return opt.recorded, sched.last_epoch


def _assert_same_grads(rec_a, rec_b):
    assert len(rec_a) == len(rec_b)
    for ga, gb in zip(rec_a, rec_b):
        rel = ((ga - gb).norm() / ga.norm()).item()
        assert rel < 1e-5, rel


def _max_param_diff(m1, m2):
    return max(
        (p1 - p2).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters())
    )


@pytest.fixture
def base_model():
    torch.manual_seed(0)
    return MS2FNet_tcn(_small_config())


@pytest.fixture(params=["clip", "no_clip"])
def clip_mode(request, monkeypatch):
    # The loss is ~1e4 here, so clipping always rescales the total gradient to
    # norm 1, which would hide a wrong micro-batch scale. "no_clip" records the
    # raw accumulated gradient; "clip" checks clipping happens once per step.
    if request.param == "no_clip":
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: None)
    return request.param


def _clone(model):
    # weight_norm modules do not support deepcopy; rebuild and load weights
    m = MS2FNet_tcn(_small_config())
    m.load_state_dict(model.state_dict())
    return m


def test_accumulation_matches_full_batch(base_model, clip_mode):
    full = _batches(8)
    rec_full, warm_full = _run(_clone(base_model), [full], grad_accum_steps=1)
    rec_accum, warm_accum = _run(
        _clone(base_model), [_slice(full, 0, 4), _slice(full, 4, 8)], grad_accum_steps=2
    )
    assert len(rec_full) == len(rec_accum) == 1  # one optimizer step
    assert warm_full == warm_accum == 1  # warmup counts optimizer steps
    assert rec_full[0].norm() > 0
    _assert_same_grads(rec_full, rec_accum)


def test_short_last_group_matches_full_batches(base_model, clip_mode):
    data = _batches(12)
    # reference: a batch of 8, then a batch of 4
    rec_full, warm_full = _run(
        _clone(base_model),
        [_slice(data, 0, 8), _slice(data, 8, 12)],
        grad_accum_steps=1,
    )
    # three micro-batches of 4 with accum 2: groups of 2 + 1
    rec_accum, warm_accum = _run(
        _clone(base_model),
        [_slice(data, 0, 4), _slice(data, 4, 8), _slice(data, 8, 12)],
        grad_accum_steps=2,
    )
    assert warm_full == warm_accum == 2
    _assert_same_grads(rec_full, rec_accum)


def test_accum_one_steps_every_batch(base_model):
    data = _batches(8)
    rec, warm = _run(
        _clone(base_model), [_slice(data, 0, 4), _slice(data, 4, 8)], grad_accum_steps=1
    )
    assert len(rec) == warm == 2


def test_gradients_cleared_between_groups(base_model, clip_mode):
    # With accum 2 over 4 micro-batches, the second optimizer step must see
    # only micro-batches 3+4. The reference for each group is a separate
    # train_step call (which starts from a fresh model, so it cannot share a
    # carry-over bug with the run under test).
    data = _batches(16)
    rec_accum, _ = _run(
        _clone(base_model),
        [_slice(data, i, i + 4) for i in range(0, 16, 4)],
        grad_accum_steps=2,
    )
    ref = [
        _run(_clone(base_model), [_slice(data, a, a + 8)], grad_accum_steps=1)[0][0]
        for a in (0, 8)
    ]
    _assert_same_grads(ref, rec_accum)
