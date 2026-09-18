# Raw dimensions as the dictionary

Paper: app:depth (raw)

Run the causal experiments directly on flagged RAW dimensions (identity dictionary)
at the final layer and layer 11; shows no causal selectivity.

```
python -m causal_analysis.common.steer_expansion --family vjepa2_raw     --stage cache
python -m causal_analysis.common.steer_expansion --family vjepa2_raw     --stage amp_allclass
python -m causal_analysis.common.steer_expansion --family vjepa2_raw     --stage erasure_allclass
python -m causal_analysis.common.steer_expansion --family vjepa2_raw_l11 --stage erasure_allclass
```
FAMILY['vjepa2_raw*'] set sae='identity' (IdentitySAE).
