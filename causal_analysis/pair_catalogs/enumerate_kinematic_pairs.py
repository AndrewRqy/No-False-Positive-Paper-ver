"""SSv2 class pairs that differ by exactly one temporal kinematic concept.

Kinematic concepts: direction of motion (horizontal, vertical, depth, or
relative approach/separation), speed, and acceleration (speeding up, slowing,
stopping, or continuing). Excludes temporal-order reversals (open/close),
action-completion pairs (pretend/fail), and pure outcome/degree contrasts
(stack collapses/holds, tear a little/in two) that are not a single kinematic
variable.

Each pair is marked clean (appearance otherwise matched, a single-frame model
could not separate them) or confounded (the differing kinematic event also
changes the object's end configuration, so appearance shifts too).
"""

import json

LABELS = json.load(
    open(
        "../SSv2/raw/20bn-something-something-download-package-labels/labels/labels.json",
        encoding="utf-8",
    )
)
IDX2NAME = (
    {int(i): n for n, i in LABELS.items()}
    if isinstance(LABELS, dict)
    else {i: n for i, n in enumerate(LABELS)}
)

# (a, b, key, concept, clean?)
PAIRS = [
    # ---- DIRECTION ----
    (86, 87, "pull_lr", "direction: horizontal", True),
    (93, 94, "push_lr", "direction: horizontal", True),
    (166, 167, "cam_lr", "direction: horizontal", True),
    (165, 168, "cam_ud", "direction: vertical", True),
    (43, 45, "move_ud", "direction: vertical", True),
    (41, 44, "obj_cam", "direction: depth", True),
    (40, 42, "obj_toaway", "direction: relative depth", True),
    (36, 37, "twoobj_apart", "direction: relative", True),
    (0, 32, "cam_appr", "direction: depth (camera)", True),
    (23, 24, "roll_downup", "direction: vertical (slope)", False),
    # ---- SPEED ----
    (134, 135, "fall_speed", "speed", True),
    (55, 56, "poke_move", "speed: impulse magnitude", True),
    # ---- ACCELERATION (clean: deceleration/continuation, appearance matched) ----
    (139, 140, "spin_stop", "acceleration: decelerate/stop", True),
    (132, 133, "collide_halt", "acceleration: deflect/halt", True),
    # ---- ACCELERATION (a fall/slide/drop event occurs or not; end config shifts) ----
    (27, 28, "lift_drop", "acceleration: drop event", False),
    (30, 31, "liftend_drop", "acceleration: drop event", False),
    (25, 26, "liftsurf_slide", "acceleration: slide event", False),
    (98, 99, "push_falloff", "acceleration: fall event", False),
    (34, 35, "moveacross_fall", "acceleration: fall event", False),
    (156, 157, "tilt_fall", "acceleration: fall event", False),
    (115, 116, "slant_slide", "acceleration: slide event", False),
    (153, 154, "throw_catch", "acceleration: catch/fall", False),
]


def main():
    for a, b, key, concept, clean in PAIRS:
        assert a in IDX2NAME and b in IDX2NAME, key

    clean = [p for p in PAIRS if p[4]]
    conf = [p for p in PAIRS if not p[4]]
    for title, group in [
        ("CLEAN single-concept (appearance matched)", clean),
        ("CONFOUNDED (kinematic event also shifts appearance)", conf),
    ]:
        print(f"\n=== {title} ({len(group)}) ===")
        for a, b, key, concept, _ in group:
            print(
                f"  {key:16s} {concept:28s} [{a:3d}] {IDX2NAME[a][:34]:36s}| [{b:3d}] {IDX2NAME[b][:34]}"
            )

    from collections import Counter

    byc = Counter(p[3].split(":")[0] for p in PAIRS)
    print(f"\nby concept: " + ", ".join(f"{k} {v}" for k, v in byc.items()))
    print(f"clean single-concept pairs: {len(clean)}")
    print(f"+ confounded kinematic-event pairs: {len(conf)}  ->  total {len(PAIRS)}")

    flat = [
        {
            "key": k,
            "concept": c,
            "clean": cl,
            "pos_idx": a,
            "neg_idx": b,
            "pos": IDX2NAME[a],
            "neg": IDX2NAME[b],
        }
        for a, b, k, c, cl in PAIRS
    ]
    json.dump(flat, open("analysis/kinematic_pairs.json", "w"), indent=1)
    print(f"saved -> analysis/kinematic_pairs.json")


if __name__ == "__main__":
    main()
