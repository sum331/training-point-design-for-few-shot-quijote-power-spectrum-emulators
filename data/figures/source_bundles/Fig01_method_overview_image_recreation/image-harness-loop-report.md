# Harness Loop Report

- selected candidate: `byte_exact_renderer`
- seed: `7`
- python: `3.11.9`

## Candidate Scores

| candidate | utility | weighted score | pass rate | avg latency ms | failures |
|---|---:|---:|---:|---:|---:|
| byte_exact_renderer | 0.9997 | 1.0000 | 100.00% | 76.57 | 0 |
| pil_resave_renderer | 0.4865 | 0.5000 | 60.00% | 242.25 | 2 |
| one_pixel_perturbation | 0.0747 | 0.1250 | 20.00% | 225.52 | 4 |

## Failures

### pil_resave_renderer
- `sha256_exact`: exact mismatch; output='be89a536c9787324009700d9e86aad20aa2207d826a84a7ac6fea324efaca177'; expected='b3d8fb4d2b9da11f3217d3e2da8e198ecbc7a42519cc1b4d7681596aa8e387ba'
- `byte_equal`: exact mismatch; output=False; expected=True

### one_pixel_perturbation
- `sha256_exact`: exact mismatch; output='0f2875d653fbf32a292658d96a90ae24bf43c3c0055af214f97607c9c9ca33af'; expected='b3d8fb4d2b9da11f3217d3e2da8e198ecbc7a42519cc1b4d7681596aa8e387ba'
- `byte_equal`: exact mismatch; output=False; expected=True
- `max_pixel_delta_zero`: outside tolerance; output=1; expected=0
- `mse_zero`: outside tolerance; output=2.1206191189930283e-07; expected=0

