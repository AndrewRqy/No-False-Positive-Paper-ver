# Flipping, randomly drawn pairs

Paper: tab:flippers_rnd

Selection-bias control: repeat the flip screen on 10 randomly drawn opposite-motion
pairs (7 temporal + 3 control, draw seed 0).

```
python -m causal_analysis.exp07_flip_random_pairs.gen_random_pairs   # -> random_pairs.json
python -m causal_analysis.common.steer_expansion --family videomae --stage pairs --pairs_json <random_pairs.json> --pairs_tag rnd
```
