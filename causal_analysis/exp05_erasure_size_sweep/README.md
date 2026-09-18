# Top-k span-size erasure sweep

Paper: tab:erasure_size

Erase the top-k identified features (k = N/4, N/2, N) vs size-matched random spans.

```
python -m causal_analysis.exp05_erasure_size_sweep.steer_erasure_size_sweep --ks 12 25 50 109
```
Needs SAE, `sae_nfp_v2.pt`, `videomae_erasure.json`, SSv2.
