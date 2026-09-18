"""Enumerate all SSv2 class pairs distinguishable only by temporal information.

A pair qualifies if telling the two classes apart requires the temporal
profile of the motion (direction, speed, acceleration, temporal order, or
action completion) rather than static appearance. Grouped by the kind of
temporal contrast so the boundary can be chosen.
"""
import json

LABELS = json.load(open(
    "../SSv2/raw/20bn-something-something-download-package-labels/labels/labels.json",
    encoding="utf-8"))
NAME2IDX = {n: int(i) for n, i in LABELS.items()} if isinstance(LABELS, dict) \
    else {n: i for i, n in enumerate(LABELS)}
IDX2NAME = {i: n for n, i in NAME2IDX.items()}

# (index_a, index_b, short key). Category dict below.
PAIRS = {
    "direction": [  # opposite spatial motion / camera direction, matched appearance
        (86, 87, "pull_lr"), (93, 94, "push_lr"),
        (166, 167, "cam_lr"), (165, 168, "cam_ud"),
        (43, 45, "move_ud"), (41, 44, "obj_cam"),
        (40, 42, "obj_toaway"), (36, 37, "twoobj_apart"),
        (0, 32, "cam_appr"),
    ],
    "order": [  # reversed temporal order, similar appearance (time-reversal pairs)
        (46, 5, "open_close"), (6, 171, "cover_uncover"),
        (14, 172, "fold_unfold"), (49, 89, "plug_unplug"),
        (106, 148, "putin_takeout"), (112, 95, "puton_pushoff"),
        (47, 109, "pick_put"), (147, 109, "take_put"),
        (121, 6, "reveal_cover"),
    ],
    "speed": [  # same event, different speed / manner
        (134, 135, "fall_speed"), (23, 24, "roll_downup"),
    ],
    "accel_outcome": [  # acceleration / completion / outcome of the motion
        (139, 140, "spin_stop"), (27, 28, "lift_drop"),
        (30, 31, "liftend_drop"), (25, 26, "liftsurf_slide"),
        (98, 99, "push_falloff"), (53, 54, "stack_collapse"),
        (34, 35, "moveacross_fall"), (156, 157, "tilt_fall"),
        (115, 116, "slant_slide"), (132, 133, "collide_halt"),
        (91, 92, "stretch_separate"), (90, 91, "pull_nothing_stretch"),
        (149, 150, "tear_amount"), (55, 56, "poke_move"),
        (57, 58, "poke_fall_spin"), (50, 49, "plug_pullout"),
        (153, 154, "throw_catch"),
    ],
    "pretend": [  # real action vs pretended (aborted / incomplete motion)
        (46, 67, "open_pretend"), (5, 66, "close_pretend"),
        (47, 68, "pick_pretend"), (151, 83, "throw_pretend"),
        (62, 70, "pour_pretend"), (147, 81, "take_pretend"),
        (148, 82, "takeout_pretend"), (106, 72, "putin_pretend"),
        (104, 71, "putbehind_pretend"), (107, 73, "putnext_pretend"),
        (109, 74, "putsurf_pretend"), (112, 75, "putonto_pretend"),
        (118, 76, "putunder_pretend"), (164, 84, "turn_pretend"),
        (143, 80, "squeeze_pretend"), (123, 77, "scoop_pretend"),
    ],
    "fail": [  # successful action vs failing partway
        (1, 161, "attach_fail"), (106, 13, "putin_fail"),
        (59, 163, "pour_fail"), (170, 64, "twist_fail"),
        (173, 63, "wipe_fail"), (2, 162, "bend_fail"),
    ],
}

# kinematic tiers: direction+speed+accel are the NFP-concept-adjacent pairs;
# order adds temporal-reversal; pretend+fail add higher-level completion
KINEMATIC = ["direction", "speed", "accel_outcome"]
REVERSAL = ["order"]
COMPLETION = ["pretend", "fail"]


def main():
    # validate every index exists
    for cat, pairs in PAIRS.items():
        for a, b, key in pairs:
            assert a in IDX2NAME and b in IDX2NAME, f"bad idx in {key}"

    total = 0
    for cat, pairs in PAIRS.items():
        print(f"\n=== {cat} ({len(pairs)}) ===")
        for a, b, key in pairs:
            print(f"  {key:22s} [{a:3d}] {IDX2NAME[a][:44]:46s} | [{b:3d}] {IDX2NAME[b][:44]}")
        total += len(pairs)

    kin = sum(len(PAIRS[c]) for c in KINEMATIC)
    rev = sum(len(PAIRS[c]) for c in REVERSAL)
    comp = sum(len(PAIRS[c]) for c in COMPLETION)
    print(f"\n{'='*60}")
    print(f"kinematic (direction+speed+accel/outcome): {kin}")
    print(f"+ temporal-order reversal:                 {rev}  -> {kin+rev}")
    print(f"+ action completion (pretend+fail):        {comp}  -> {total}")
    print(f"TOTAL temporally-distinguished pairs:      {total}")

    # dump flat list for downstream screening
    flat = []
    for cat, pairs in PAIRS.items():
        for a, b, key in pairs:
            flat.append({"key": key, "category": cat,
                         "pos_idx": a, "neg_idx": b,
                         "pos": IDX2NAME[a], "neg": IDX2NAME[b]})
    json.dump(flat, open("analysis/temporal_pairs_all.json", "w"), indent=1)
    print(f"\nsaved {len(flat)} pairs -> analysis/temporal_pairs_all.json")


if __name__ == "__main__":
    main()
