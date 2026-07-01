# Harness Loop Report

- selected candidate: `byte_exact_renderer`
- seed: `7`
- python: `3.11.9`

## Candidate Scores

| candidate | utility | weighted score | pass rate | avg latency ms | failures |
|---|---:|---:|---:|---:|---:|
| byte_exact_renderer | 0.9997 | 1.0000 | 100.00% | 73.74 | 0 |
| pil_resave_renderer | 0.4864 | 0.5000 | 60.00% | 263.74 | 2 |
| one_pixel_perturbation | 0.0746 | 0.1250 | 20.00% | 263.12 | 4 |

## Failures

### pil_resave_renderer
- `sha256_exact`: exact mismatch; output='c55c18addd2d348d49f9b48004b7216d3772fe7ad01c5e1437112d2ac1778336'; expected='866709bb96109e975ceae6203d8e7f463fb80bd2ddcf136c77936e06841902ef'
- `byte_equal`: exact mismatch; output=False; expected=True

### one_pixel_perturbation
- `sha256_exact`: exact mismatch; output='06d7490e2eedf6165ec81a0615f718968cbec283f78401bda3c90ada01261b84'; expected='866709bb96109e975ceae6203d8e7f463fb80bd2ddcf136c77936e06841902ef'
- `byte_equal`: exact mismatch; output=False; expected=True
- `max_pixel_delta_zero`: outside tolerance; output=1; expected=0
- `mse_zero`: outside tolerance; output=2.1195915970910726e-07; expected=0

