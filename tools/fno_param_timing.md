# FNO vs UNet ¡ª parameter count & inference timing

Device: `cuda`  |  input `(B,43,128,128)`  |  batch 8, 30 iters

| model | params | ¦¤params vs baseline | ms/sample | imgs/sec |
|---|---|---|---|---|
| baseline | 14,453,649 | +0 (+0.00%) | 0.337 | 2967.3 |
| fno_bottleneck | 17,862,033 | +3,408,384 (+23.58%) | 0.317 | 3152.6 |
| fno_16x16 | 16,567,313 | +2,113,664 (+14.62%) | 0.330 | 3027.8 |
