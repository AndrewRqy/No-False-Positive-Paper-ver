# Amplification at alpha=3, four views

Paper: tab:repair

The four columns (all-class / top-10 / bottom-10 / feature-dependent-class accuracy)
are SLICES at alpha=3 of the per-class accuracies written by `amp_allclass`; there is
no separate script.

```
python -m causal_analysis.common.steer_expansion --family videomae --stage amp_allclass --alphas 1.5 2 3 4 5 6 8 10
```
Output: `<fam>_amp_allclass.json` -> read the alpha=3 entry; `net_acc` is the all-class
view, and rank `per_class` to form top-10 / bottom-10; the feature-dependent classes
are those in `<fam>_erasure.json`.
