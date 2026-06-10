"""CPU-only validation of the Phase 3 UTAE wiring. No GPU, no dataset I/O.
Confirms: (1) get_n_features == 42 for no-PDSI T=5 non-dedup;
(2) UTAELightning builds with that channel count;
(3) a fake (B,T,C,H,W) batch + doys forward-passes to (B,1,H,W)."""
import sys
sys.path.insert(0, 'src')
import torch
from dataloader.FireSpreadDataset import FireSpreadDataset
from models.UTAELightning import UTAELightning

NO_PDSI_KEEP = [i for i in range(43) if i != 15]

n = FireSpreadDataset.get_n_features(5, NO_PDSI_KEEP, False)
print(f'[check] get_n_features(5, no_pdsi, dedup=False) = {n}', flush=True)
assert n == 42, f'expected 42 per-timestep channels, got {n}'

# also confirm all-features path = 43 (sanity)
n_all = FireSpreadDataset.get_n_features(5, None, False)
print(f'[check] get_n_features(5, all, dedup=False) = {n_all}', flush=True)
assert n_all == 43, f'expected 43, got {n_all}'

model = UTAELightning(
    n_channels=n,
    flatten_temporal_dimension=False, pos_class_weight=236.0,
    loss_function='Focal',  # use_doy is hardcoded True inside UTAELightning
).eval()
nparams = sum(p.numel() for p in model.parameters())
print(f'[check] UTAELightning built, params = {nparams/1e6:.2f}M', flush=True)

B, T, C, H, W = 2, 5, n, 128, 128
x = torch.randn(B, T, C, H, W)
doys = torch.randint(1, 365, (B, T)).float()
with torch.no_grad():
    out = model(x, doys)
print(f'[check] forward out shape = {tuple(out.shape)}', flush=True)
assert out.shape[0] == B and out.shape[1] == 1 and out.shape[-2:] == (H, W), \
    f'unexpected output shape {tuple(out.shape)}'

# loss path: target (B,H,W) long
y = (torch.rand(B, H, W) > 0.99).long()
yhat = out.squeeze(1)
loss = model.compute_loss(yhat, y)
print(f'[check] focal loss on fake batch = {loss.item():.6f}', flush=True)
print('[check] ALL PASSED', flush=True)
