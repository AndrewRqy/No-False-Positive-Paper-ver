# Binary-discrimination span erasure (NOT in paper)

Paper: -- supplementary --

NOT part of the paper (kept per request). Restrict pre-softmax logits to a class
pair, softmax over the two, and measure the binary-accuracy drop when the identified
SAE span is erased, vs 3 size-matched random feature sets. Reference appearance-
separable pairs test specificity.

```
python -m causal_analysis.binary_discrimination_erasure.pair_binary_erasure \
    --family videomae --pairs_json binary_pairs.json \
    --ssv2_videos <...> --ssv2_val_json <.../validation.json>
```
Pair catalog: `binary_pairs.json` here; enumerators in `causal_analysis/pair_catalogs/`.
