"""Unit tests + data-independent deliverables for the FNO model.

Checks (no dataset needed):
  1. SpectralConv2d / FNOBlock forward on dummy bottleneck tensor: shape + no NaN.
  2. Full FNOSMPModel vs baseline SMPModel forward on (B,43,128,128): shape + no NaN.
  3. Parameter counts (UNet baseline vs FNO variants).
  4. Inference timing (ms/sample, imgs/sec) on the available device, same batch.
Writes a markdown results table to tools/fno_param_timing.md.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch
from models import SMPModel, FNOSMPModel
from models.fno_blocks import SpectralConv2d, FNOBlock

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CH = 43            # channels after sin/cos angle encoding (T=1, full features)
OUT_MD = os.path.join(os.path.dirname(__file__), 'fno_param_timing.md')


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def check(name, cond):
    print(f'  [{"OK" if cond else "FAIL"}] {name}')
    assert cond, name


def time_model(model, batch=16, iters=50, warmup=10):
    model.eval().to(DEV)
    x = torch.randn(batch, N_CH, 128, 128, device=DEV)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if DEV == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if DEV == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return dt / batch * 1000, batch / dt        # ms/sample, imgs/sec


def build(kind):
    common = dict(encoder_name='resnet18', n_channels=N_CH,
                  flatten_temporal_dimension=True, pos_class_weight=236,
                  loss_function='Focal')
    if kind == 'baseline':
        return SMPModel(**common)
    if kind == 'fno_bottleneck':
        return FNOSMPModel(fno_stage=5, fno_modes_h=2, fno_modes_w=3, n_fno_blocks=1, **common)
    if kind == 'fno_16x16':
        return FNOSMPModel(fno_stage=3, fno_modes_h=8, fno_modes_w=8, n_fno_blocks=1, **common)


def main():
    print(f'Device: {DEV}')
    print('1) SpectralConv2d / FNOBlock dummy forward:')
    x = torch.randn(2, 512, 4, 4, device=DEV)
    sc = SpectralConv2d(512, 512, 4, 4).to(DEV)
    y = sc(x); check('SpectralConv2d shape', y.shape == x.shape); check('no NaN', not torch.isnan(y).any())
    fb = FNOBlock(512, 4, 4).to(DEV)
    y = fb(x); check('FNOBlock shape', y.shape == x.shape); check('no NaN', not torch.isnan(y).any())

    print('2) Full model forward (B,43,128,128):')
    rows = []
    for kind in ['baseline', 'fno_bottleneck', 'fno_16x16']:
        m = build(kind)
        xin = torch.randn(2, N_CH, 128, 128, device=DEV)
        m.eval().to(DEV)
        with torch.no_grad():
            out = m(xin)
        check(f'{kind} out shape (2,1,128,128)', tuple(out.shape) == (2, 1, 128, 128))
        check(f'{kind} no NaN', not torch.isnan(out).any())
        np_ = count_params(m)
        ms, ips = time_model(m)
        rows.append((kind, np_, ms, ips))
        print(f'    {kind}: params={np_:,}  {ms:.3f} ms/sample  {ips:.1f} imgs/s')

    base_p = rows[0][1]
    with open(OUT_MD, 'w') as f:
        f.write('# FNO vs UNet — parameter count & inference timing\n\n')
        f.write(f'Device: `{DEV}`  |  input `(B,43,128,128)`  |  batch 8, 30 iters\n\n')
        f.write('| model | params | Δparams vs baseline | ms/sample | imgs/sec |\n')
        f.write('|---|---|---|---|---|\n')
        for kind, np_, ms, ips in rows:
            d = np_ - base_p
            f.write(f'| {kind} | {np_:,} | {d:+,} ({100*d/base_p:+.2f}%) | {ms:.3f} | {ips:.1f} |\n')
    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
