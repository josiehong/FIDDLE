import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from msfiddle.optim import Muon, split_matrix_params, zeropower_via_newtonschulz5


def _toy_model():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = weight_norm(nn.Conv1d(4, 8, kernel_size=3, padding=1))
            self.fc = weight_norm(nn.Linear(8, 5))
            self.emb = weight_norm(nn.Embedding(7, 4))
            self.bias_only = nn.Linear(5, 5)

        def forward(self, x, idx):
            h = self.conv(x).mean(dim=2)
            return self.fc(h) + self.emb(idx).sum(dim=1, keepdim=True)[:, :, 0]

    return Toy()


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(32, 64)
    X = zeropower_via_newtonschulz5(G)
    assert X.shape == G.shape
    s = torch.linalg.svdvals(X.float())
    # quintic NS lands singular values near 1, not exactly 1
    assert s.min() > 0.3 and s.max() < 1.6


def test_newton_schulz_tall_matrix():
    torch.manual_seed(0)
    G = torch.randn(64, 16)
    X = zeropower_via_newtonschulz5(G)
    assert X.shape == G.shape
    s = torch.linalg.svdvals(X.float())
    assert s.min() > 0.3 and s.max() < 1.6


def test_split_matrix_params():
    model = _toy_model()
    muon_params, adamw_params = split_matrix_params(model)
    muon_ids = {id(p) for p in muon_params}
    named = dict(model.named_parameters())
    # matrix weights of conv/linear go to muon
    assert id(named["conv.weight_v"]) in muon_ids
    assert id(named["fc.weight_v"]) in muon_ids
    assert id(named["bias_only.weight"]) in muon_ids
    # gains, biases, embeddings stay on adamw
    for name in [
        "conv.weight_g",
        "fc.weight_g",
        "conv.bias",
        "fc.bias",
        "emb.weight_v",
        "emb.weight_g",
        "bias_only.bias",
    ]:
        assert id(named[name]) not in muon_ids, name
    assert len(muon_params) + len(adamw_params) == len(named)


def test_muon_reduces_loss():
    torch.manual_seed(0)
    model = _toy_model()
    opt = Muon(*split_matrix_params(model), lr=0.02, adamw_lr=1e-3)
    x = torch.randn(16, 4, 10)
    idx = torch.randint(0, 7, (16, 2))
    y = torch.randn(16, 5)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x, idx), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9


def test_muon_state_dict_roundtrip():
    torch.manual_seed(0)
    model = _toy_model()
    opt = Muon(*split_matrix_params(model), lr=0.02, adamw_lr=1e-3)
    x, idx = torch.randn(4, 4, 10), torch.randint(0, 7, (4, 2))
    loss = model(x, idx).sum()
    loss.backward()
    opt.step()
    state = opt.state_dict()
    opt2 = Muon(*split_matrix_params(model), lr=0.02, adamw_lr=1e-3)
    opt2.load_state_dict(state)
    assert len(opt2.state_dict()["state"]) == len(state["state"])
