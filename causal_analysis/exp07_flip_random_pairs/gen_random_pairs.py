"""Random Pairs - random draw of class pairs for the pair-flipping experiment.

Procedure (fixed 7 temporal + 3 control structure): repeatedly draw one class
uniformly at random from the SSv2 label list. If the drawn class belongs to a
candidate opposite pair (curated below), and that pair is unused, and its
bucket (temporal or control) is not full, the pair is added. The draw stops at
7 temporal and 3 control pairs.

The candidate map lists every template pair we judge to be opposites:
- temporal: the two classes differ by in-plane motion direction, speed, or
  acceleration profile (the NFP concept axes).
- control: the two classes differ by depth-axis motion or by static position
  (in front / behind), with matched in-plane kinematics.
"""

import argparse
import json
import random

# (key, axis, positive template, negative template, type)
CANDIDATES = [
    (
        "push_lr",
        "vel_x",
        "Pushing [something] from left to right",
        "Pushing [something] from right to left",
        "temporal",
    ),
    (
        "pull_lr",
        "vel_x",
        "Pulling [something] from left to right",
        "Pulling [something] from right to left",
        "temporal",
    ),
    (
        "cam_lr",
        "vel_x",
        "Turning the camera left while filming [something]",
        "Turning the camera right while filming [something]",
        "temporal",
    ),
    ("move_ud", "vel_y", "Moving [something] up", "Moving [something] down", "temporal"),
    (
        "cam_ud",
        "vel_y",
        "Turning the camera upwards while filming [something]",
        "Turning the camera downwards while filming [something]",
        "temporal",
    ),
    (
        "fall_speed",
        "speed",
        "[Something] falling like a rock",
        "[Something] falling like a feather or paper",
        "temporal",
    ),
    (
        "spin_stop",
        "accel_mag",
        "Spinning [something] so it continues spinning",
        "Spinning [something] that quickly stops spinning",
        "temporal",
    ),
    (
        "lift_drop",
        "accel_mag",
        "Lifting [something] up completely without letting it drop down",
        "Lifting [something] up completely, then letting it drop down",
        "temporal",
    ),
    (
        "liftend_drop",
        "accel_mag",
        "Lifting up one end of [something] without letting it drop down",
        "Lifting up one end of [something], then letting it drop down",
        "temporal",
    ),
    (
        "roll_updown",
        "vel_x",
        "Letting [something] roll down a slanted surface",
        "Letting [something] roll up a slanted surface, so it rolls back down",
        "temporal",
    ),
    (
        "throw_catch",
        "accel_mag",
        "Throwing [something] in the air and catching it",
        "Throwing [something] in the air and letting it fall",
        "temporal",
    ),
    (
        "push_falloff",
        "accel_mag",
        "Pushing [something] so that it almost falls off but doesn't",
        "Pushing [something] so that it falls off the table",
        "temporal",
    ),
    (
        "stack_collapse",
        "accel_mag",
        "Poking a stack of [something] so the stack collapses",
        "Poking a stack of [something] without the stack collapsing",
        "temporal",
    ),
    (
        "collide_halt",
        "accel_mag",
        "[Something] colliding with [something] and both are being deflected",
        "[Something] colliding with [something] and both come to a halt",
        "temporal",
    ),
    (
        "tilt_fall",
        "accel_mag",
        "Tilting [something] with [something] on it slightly so it doesn't fall down",
        "Tilting [something] with [something] on it until it falls off",
        "temporal",
    ),
    (
        "lift_slide",
        "accel_mag",
        "Lifting a surface with [something] on it but not enough for it to slide down",
        "Lifting a surface with [something] on it until it starts sliding down",
        "temporal",
    ),
    (
        "slant_slide",
        "vel_x",
        "Putting [something] that can't roll onto a slanted surface, so it slides down",
        "Putting [something] that can't roll onto a slanted surface, so it stays where it is",
        "temporal",
    ),
    (
        "move_apart",
        "vel_x",
        "Moving [something] and [something] closer to each other",
        "Moving [something] and [something] away from each other",
        "temporal",
    ),
    (
        "poke_move",
        "speed",
        "Poking [something] so it slightly moves",
        "Poking [something] so lightly that it doesn't or almost doesn't move",
        "temporal",
    ),
    (
        "obj_cam",
        "control",
        "Moving [something] towards the camera",
        "Moving [something] away from the camera",
        "control",
    ),
    (
        "obj_obj",
        "control",
        "Moving [something] closer to [something]",
        "Moving [something] away from [something]",
        "control",
    ),
    (
        "cam_appr",
        "control",
        "Approaching [something] with your camera",
        "Moving away from [something] with your camera",
        "control",
    ),
    (
        "hold_fb",
        "control",
        "Holding [something] in front of [something]",
        "Holding [something] behind [something]",
        "control",
    ),
    (
        "drop_fb",
        "control",
        "Dropping [something] in front of [something]",
        "Dropping [something] behind [something]",
        "control",
    ),
    (
        "put_fb",
        "control",
        "Putting [something] in front of [something]",
        "Putting [something] behind [something]",
        "control",
    ),
]


def main() -> None:
    """Draw the seeded temporal and control class pairs from the candidate map and save JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_temporal", type=int, default=7)
    ap.add_argument("--n_control", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    labels = sorted(json.load(open(args.labels_json)))

    def norm(t: str) -> str:
        """Normalize a template for matching: strip the "[...]" brackets and lowercase."""
        # labels.json uses bare "something"; the pair templates use "[something]"
        return t.replace("[Something]", "Something").replace("[something]", "something").lower()

    norm_labels = {norm(l): l for l in labels}
    by_label = {}
    for cand in CANDIDATES:
        for t in (cand[2], cand[3]):
            assert norm(t) in norm_labels, f"not an SSv2 template: {t}"
            by_label[norm_labels[norm(t)]] = cand

    rng = random.Random(args.seed)
    picked, used_keys = {"temporal": [], "control": []}, set()
    n_draws = 0
    while len(picked["temporal"]) < args.n_temporal or len(picked["control"]) < args.n_control:
        n_draws += 1
        cls = rng.choice(labels)
        cand = by_label.get(cls)
        if cand is None or cand[0] in used_keys:
            continue
        typ = cand[4]
        cap = args.n_temporal if typ == "temporal" else args.n_control
        if len(picked[typ]) >= cap:
            continue
        picked[typ].append(cand)
        used_keys.add(cand[0])

    pairs = [list(c[:4]) for c in picked["temporal"] + picked["control"]]
    json.dump(
        {"seed": args.seed, "n_draws": n_draws, "pairs": pairs}, open(args.output, "w"), indent=2
    )
    print(f"seed={args.seed} draws={n_draws}")
    for c in picked["temporal"]:
        print(f"  temporal {c[0]} ({c[1]})")
    for c in picked["control"]:
        print(f"  control  {c[0]}")


if __name__ == "__main__":
    main()
