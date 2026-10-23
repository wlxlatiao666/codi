# Slow local cutpoint search

Final policy (k, numerator/6): [[3, 2], [4, 2], [5, 3]]

| Dataset | Fixed control | Adaptive | Difference (pp) |
|---|---:|---:|---:|
| test | 43.52% | 42.23% | -1.29 |
| gsm-hard | 10.16% | 9.17% | -0.99 |
| multi-arith-full | 90.50% | 91.00% | +0.50 |
| svamp | 42.90% | 42.30% | -0.60 |

Single seed11. Search used validation only. No claim of a globally optimal fixed policy.
The learned policy was not retrained from scratch. See protocol.json and round-*-selection.json.
