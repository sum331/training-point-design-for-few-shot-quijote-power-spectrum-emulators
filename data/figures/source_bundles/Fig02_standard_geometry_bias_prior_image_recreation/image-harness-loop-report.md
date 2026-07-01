# Harness Loop Report

- selected candidate: `byte_exact_renderer`
- seed: `7`
- python: `3.11.9`

## Candidate Scores

| candidate | utility | weighted score | pass rate | avg latency ms | failures |
|---|---:|---:|---:|---:|---:|
| byte_exact_renderer | 0.9997 | 1.0000 | 100.00% | 73.17 | 0 |
| pil_resave_renderer | 0.4864 | 0.5000 | 60.00% | 268.65 | 2 |
| one_pixel_perturbation | 0.0746 | 0.1250 | 20.00% | 264.60 | 4 |

## Failures

### pil_resave_renderer
- `sha256_exact`: exact mismatch; output='966943fbf64acca5a723aa0ec49b92f06a3ddf81a8e43063e71c69a29c4614e2'; expected='8685cec092b8cd61a93d2ff64b00f7f5e52343781b31b6c463f61bf506bc9f0a'
- `byte_equal`: exact mismatch; output=False; expected=True

### one_pixel_perturbation
- `sha256_exact`: exact mismatch; output='a47a45e7c64970b8fa9f976a6de01f2635bb0802b42c2318d505fe413bb3a6b9'; expected='8685cec092b8cd61a93d2ff64b00f7f5e52343781b31b6c463f61bf506bc9f0a'
- `byte_equal`: exact mismatch; output=False; expected=True
- `max_pixel_delta_zero`: outside tolerance; output=1; expected=0
- `mse_zero`: outside tolerance; output=2.1192762586805554e-07; expected=0

