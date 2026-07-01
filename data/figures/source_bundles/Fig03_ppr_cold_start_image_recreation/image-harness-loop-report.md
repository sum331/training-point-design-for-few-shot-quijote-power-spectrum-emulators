# Harness Loop Report

- selected candidate: `byte_exact_renderer`
- seed: `7`
- python: `3.11.9`

## Candidate Scores

| candidate | utility | weighted score | pass rate | avg latency ms | failures |
|---|---:|---:|---:|---:|---:|
| byte_exact_renderer | 0.9997 | 1.0000 | 100.00% | 74.30 | 0 |
| pil_resave_renderer | 0.4864 | 0.5000 | 60.00% | 252.76 | 2 |
| one_pixel_perturbation | 0.0746 | 0.1250 | 20.00% | 250.54 | 4 |

## Failures

### pil_resave_renderer
- `sha256_exact`: exact mismatch; output='e42b82cd57300a7c8024c660f14c186e3191e1a8eee5fadace30fb4e1b895051'; expected='68ed6f0e11d6f7114741725553a39673bbf6889cf808068cee3590af5822d40f'
- `byte_equal`: exact mismatch; output=False; expected=True

### one_pixel_perturbation
- `sha256_exact`: exact mismatch; output='046ede747d851f03709aaf0955fbda871b42d0a41a5763da621f482dcc3b0dbd'; expected='68ed6f0e11d6f7114741725553a39673bbf6889cf808068cee3590af5822d40f'
- `byte_equal`: exact mismatch; output=False; expected=True
- `max_pixel_delta_zero`: outside tolerance; output=1; expected=0
- `mse_zero`: outside tolerance; output=2.119793786460453e-07; expected=0

