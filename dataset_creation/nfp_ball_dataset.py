"""NFP Ball Dataset - No-False-Positives stimulus set for temporal feature testing.

Essential tools (coordinate math, profile generators, trajectory computation).
The Kubric scene builder and generation loop are in generate_video() below,
but the main() entry point is not yet wired - run this file directly for a
sanity check of the tool functions.

Dataset design (see nfp_dataset_plan.tex):
  - N = 3000 videos, each with independently sampled start position and velocity profile
  - Start position: Uniform[SPAWN_MIN_M, SPAWN_MAX_M]^2  (coverage condition)
  - Velocity profile: Family A (speed-varying, fixed dir) or Family B (dir-varying, fixed speed)
  - Analytic trajectory written as Blender keyframes; no PyBullet simulation
  - All tau values and token indices pre-computed and stored in metadata.json

Usage (inside kubric Docker container):
  python -m dataset_creation.nfp_ball_dataset --sanity_check        # run tool tests only
  python -m dataset_creation.nfp_ball_dataset \\
      --output_dir /output/nfp \\
      --n_videos 3000 \\
      --start_idx 0 --end_idx 999   # shard for parallel jobs
"""

import argparse
import json
import math
import pathlib
import shutil
import tempfile
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scene / VideoMAE constants
# ---------------------------------------------------------------------------
CANVAS_PX = 224  # VideoMAE native resolution (px)
FIELD_M = 14.0  # visible field width in world space (m), centred at origin
HALF_FIELD_M = FIELD_M / 2  # 7.0 m
PX_PER_M = CANVAS_PX / FIELD_M  # 16.0 px / m
M_PER_PX = FIELD_M / CANVAS_PX  # 0.0625 m / px

FRAME_RATE = 24  # fps
T = 16  # frames per video
N_TUBELETS = T // 2  # 8 temporal steps (2 frames each)
PATCH_PX = 16  # VideoMAE spatial patch size (px)
N_PATCHES = CANVAS_PX // PATCH_PX  # 14 patches per axis → 196 tokens total

BALL_RADIUS_M = 0.9  # ~14.4 px radius at 224 px resolution
CAMERA_Z = 20.0  # orthographic camera height (m); only z matters
KUBRIC_HALF_M = 12.0  # half-width of Kubric scene (must exceed spawn range)

# ---------------------------------------------------------------------------
# Coverage rectangle  (exact minimum from coverage condition)
# ---------------------------------------------------------------------------
V_MAX_MPS = 3.5  # maximum base speed (m/s)
V_MIN_MPS = 0.5  # minimum base speed (m/s)
S_MAX_M = V_MAX_MPS * T / FRAME_RATE  # 2.333 m  - worst-case axis displacement
SPAWN_MIN_M = -(HALF_FIELD_M + S_MAX_M)  # -9.333 m
SPAWN_MAX_M = HALF_FIELD_M + S_MAX_M  #  9.333 m

# ---------------------------------------------------------------------------
# Profile type registries
# ---------------------------------------------------------------------------
FAMILY_A_TYPES = [
    "constant",
    "linear_accel",
    "linear_decel",
    "sinusoidal",
    "slow_fast_slow",
    "fast_slow_fast",
    "step_accel",
    "step_decel",
]

FAMILY_B_TYPES = [
    "gradual_turn",
    "sharp_turn",
    "back_and_forth",
]


# ===========================================================================
# 1.  COORDINATE CONVERSION
# ===========================================================================


def world_to_px(world_x: float, world_y: float) -> Tuple[float, float]:
    """Convert world-space metres to pixel coordinates.

    World space is metres with the origin at the scene centre and y-up; pixel
    space has its origin at the top-left corner with y-down.

    Derivation:
      px_x = (world_x - (-7)) / 14 * 224 = (world_x + 7) * 16
      px_y = (7 - world_y)  / 14 * 224  = (7 - world_y) * 16   [y-flip]

    Args:
        world_x: X coordinate in world metres.
        world_y: Y coordinate in world metres.

    Returns:
        The (px_x, px_y) pixel coordinates.
    """
    px_x = (world_x + HALF_FIELD_M) * PX_PER_M
    px_y = (HALF_FIELD_M - world_y) * PX_PER_M
    return px_x, px_y


def px_to_world(px_x: float, px_y: float) -> Tuple[float, float]:
    """Inverse of world_to_px."""
    world_x = px_x * M_PER_PX - HALF_FIELD_M
    world_y = -px_y * M_PER_PX + HALF_FIELD_M
    return world_x, world_y


def is_on_screen(px_x: float, px_y: float) -> bool:
    """True if the ball centre lies within [0, CANVAS_PX)^2."""
    return 0.0 <= px_x < CANVAS_PX and 0.0 <= px_y < CANVAS_PX


def get_spatial_token(px_x: float, px_y: float) -> int:
    """Raster-order (row-major) spatial token index for a pixel's patch.

    b(V,t) = floor(px_y / 16) * 14 + floor(px_x / 16)

    Args:
        px_x: X pixel coordinate.
        px_y: Y pixel coordinate.

    Returns:
        The row-major token index of the containing patch, or -1 if off-screen.
    """
    if not is_on_screen(px_x, px_y):
        return -1
    row = int(px_y) // PATCH_PX  # 0 .. 13
    col = int(px_x) // PATCH_PX  # 0 .. 13
    return row * N_PATCHES + col


def get_temporal_step(frame: int) -> int:
    """VideoMAE tubelet index: two consecutive frames share one temporal step."""
    return frame // 2


# ===========================================================================
# 2.  SPAWN RECTANGLE SAMPLER  (S1 + S3)
# ===========================================================================


def sample_start_position(rng: np.random.Generator) -> Tuple[float, float]:
    """Draw a start position uniformly from the coverage spawn rectangle.

    Must be called independently of sample_velocity_profile to satisfy S3.

    Args:
        rng: NumPy random generator.

    Returns:
        The (x0, y0) start position in world metres.
    """
    x0 = float(rng.uniform(SPAWN_MIN_M, SPAWN_MAX_M))
    y0 = float(rng.uniform(SPAWN_MIN_M, SPAWN_MAX_M))
    return x0, y0


# ===========================================================================
# 3.  VELOCITY PROFILE GENERATORS  (S2)
# ===========================================================================

_GEO_RATE = {
    "geo_a115": 1.15,
    "geo_a130": 1.30,
    "geo_a150": 1.50,
    "geo_d087": 1.0 / 1.15,
    "geo_d077": 1.0 / 1.30,
    "geo_d067": 1.0 / 1.50,
}
GEOMETRIC_TYPES = list(_GEO_RATE.keys())


def _family_a_speeds(profile_type: str, s: float) -> np.ndarray:
    """Per-frame speed sequence for Family A (fixed direction, varying speed).

    Mean speed = s for all types except sinusoidal (mean = s/2), so cumulative
    displacement <= s * T / FRAME_RATE <= S_MAX_M for all types.

    Args:
        profile_type: One of FAMILY_A_TYPES or a geometric "geo_*" type.
        s: Base speed in m/s.

    Returns:
        A [T] array of per-frame speeds.

    Raises:
        ValueError: If profile_type is not a known Family A type.
    """
    t = np.arange(T, dtype=np.float64)
    s_lo = s / 2.0
    s_hi = 3.0 * s / 2.0

    if profile_type == "constant":
        return np.full(T, s)

    elif profile_type == "linear_accel":
        # ramps from s_lo to s_hi; mean = s
        return s_lo + (s_hi - s_lo) * t / (T - 1)

    elif profile_type == "linear_decel":
        return s_hi - (s_hi - s_lo) * t / (T - 1)

    elif profile_type == "sinusoidal":
        # mean = s/2; total displacement = s*T/(2*FRAME_RATE) < S_MAX_M ✓
        return s * (0.5 + 0.5 * np.sin(2.0 * np.pi * t / T))

    elif profile_type == "slow_fast_slow":
        # s_lo for outer quarters, s_hi for middle half; mean = s
        return np.where((t >= 4) & (t <= 11), s_hi, s_lo)

    elif profile_type == "fast_slow_fast":
        return np.where((t >= 4) & (t <= 11), s_lo, s_hi)

    elif profile_type == "step_accel":
        # s_lo first half, s_hi second half; mean = s
        return np.where(t < 8, s_lo, s_hi)

    elif profile_type == "step_decel":
        return np.where(t < 8, s_hi, s_lo)

    elif profile_type.startswith("geo_"):
        # geometric speed: s_seq[i+1]/s_seq[i] = r (constant ratio), so
        # accel_mag(i) = s_seq[i] * |r - 1| is exactly proportional to speed,
        # giving within-video corr(accel_mag, speed) = 1. Scaled so the peak
        # speed equals s (<= V_MAX), keeping displacement within field bounds.
        # Type name encodes r: geo_a130 -> r=1.30 (accel), geo_d077 -> r=0.77.
        r = _GEO_RATE[profile_type]
        base = r**t
        return base / base.max() * s

    else:
        raise ValueError(f"Unknown Family A type: {profile_type!r}")


def _family_b_directions(profile_type: str, theta: float, delta: float) -> np.ndarray:
    """Per-frame direction sequence for Family B (fixed speed, varying direction).

    Args:
        profile_type: One of FAMILY_B_TYPES.
        theta: Initial direction in radians.
        delta: Turn angle in radians (ignored for back_and_forth, which uses pi).

    Returns:
        A [T] array of per-frame directions in radians.

    Raises:
        ValueError: If profile_type is not a known Family B type.
    """
    t = np.arange(T, dtype=np.float64)

    if profile_type == "gradual_turn":
        # direction rotates linearly from theta to theta + delta
        return theta + delta * t / (T - 1)

    elif profile_type == "sharp_turn":
        # instantaneous direction change at frame 8
        return np.where(t < 8, theta, theta + delta)

    elif profile_type == "back_and_forth":
        # reverses direction at frame 8; delta is always pi
        return np.where(t < 8, theta, theta + math.pi)

    else:
        raise ValueError(f"Unknown Family B type: {profile_type!r}")


def sample_velocity_profile(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample a complete velocity profile independently of start position (S2, S3).

    Args:
        rng: NumPy random generator.

    Returns:
        A profile dict with keys:
          family        : "A" or "B"
          profile_type  : one of FAMILY_A_TYPES / FAMILY_B_TYPES
          speed_mps     : base speed s (float)
          direction_deg : initial direction theta in degrees
          delta_deg     : turn angle for Family B (omitted for back_and_forth)
          vx, vy        : np.ndarray [T] per-frame velocity components (m/s)
    """
    s = float(rng.uniform(V_MIN_MPS, V_MAX_MPS))
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    family = rng.choice(["A", "B"])

    if family == "A":
        ptype = str(rng.choice(FAMILY_A_TYPES))
        speeds = _family_a_speeds(ptype, s)
        directions = np.full(T, theta)

    else:  # Family B
        ptype = str(rng.choice(FAMILY_B_TYPES))
        if ptype == "back_and_forth":
            delta = math.pi
        else:
            delta = float(rng.uniform(math.radians(45), math.radians(180)))
        speeds = np.full(T, s)
        directions = _family_b_directions(ptype, theta, delta)

    vx = speeds * np.cos(directions)
    vy = speeds * np.sin(directions)

    profile: Dict = {
        "family": family,
        "profile_type": ptype,
        "speed_mps": round(s, 4),
        "direction_deg": round(math.degrees(theta), 3),
        "vx": vx,
        "vy": vy,
    }
    if family == "B" and ptype != "back_and_forth":
        profile["delta_deg"] = round(math.degrees(delta), 3)

    return profile


# ===========================================================================
# 4.  GROUND-TRUTH TAU COMPUTATION
# ===========================================================================


def compute_tau(frame: int, vx: np.ndarray, vy: np.ndarray) -> Dict[str, float]:
    """Compute all temporal concept (tau) values at one frame.

    Every quantity depends only on the velocity profile, not on (x0, y0),
    satisfying assumption A5 of the proof. accel_mag at the last frame is
    defined as 0 (no next frame).

    Args:
        frame: Frame index in [0, T).
        vx: [T] per-frame x velocity components.
        vy: [T] per-frame y velocity components.

    Returns:
        A dict of the tau concepts speed, vel_x, vel_y, accel_mag, direction.
    """
    vx_t = float(vx[frame])
    vy_t = float(vy[frame])
    speed = math.hypot(vx_t, vy_t)

    if frame + 1 < T:
        accel_mag = math.hypot(float(vx[frame + 1]) - vx_t, float(vy[frame + 1]) - vy_t)
    else:
        accel_mag = 0.0

    return {
        "speed": round(speed, 5),
        "vel_x": round(vx_t, 5),
        "vel_y": round(vy_t, 5),
        "accel_mag": round(accel_mag, 5),
        "direction": round(math.atan2(vy_t, vx_t), 5),
    }


# ===========================================================================
# 5.  ANALYTIC TRAJECTORY COMPUTATION
# ===========================================================================


@dataclass
class FrameState:
    frame: int
    pos_m: Tuple[float, float]
    pos_px: Tuple[float, float]
    vel_mps: Tuple[float, float]
    on_screen: bool
    spatial_token: int
    temporal_step: int
    tau: Dict[str, float]


def compute_trajectory(
    x0_m: float,
    y0_m: float,
    profile: Dict[str, Any],
) -> List[FrameState]:
    """Compute the full T-frame trajectory analytically.

    Position advances by v(t) / FRAME_RATE each frame. No physics engine is
    used; positions are exact.

    Args:
        x0_m: Start x position in world metres.
        y0_m: Start y position in world metres.
        profile: Velocity profile dict with "vx" and "vy" arrays.

    Returns:
        The list of FrameState objects, one per frame.
    """
    vx: np.ndarray = profile["vx"]
    vy: np.ndarray = profile["vy"]

    states: List[FrameState] = []
    x_m, y_m = x0_m, y0_m

    for frame in range(T):
        px_x, px_y = world_to_px(x_m, y_m)
        on_scr = is_on_screen(px_x, px_y)

        states.append(
            FrameState(
                frame=frame,
                pos_m=(round(x_m, 5), round(y_m, 5)),
                pos_px=(round(px_x, 2), round(px_y, 2)),
                vel_mps=(round(float(vx[frame]), 5), round(float(vy[frame]), 5)),
                on_screen=on_scr,
                spatial_token=get_spatial_token(px_x, px_y),
                temporal_step=get_temporal_step(frame),
                tau=compute_tau(frame, vx, vy),
            )
        )

        x_m += float(vx[frame]) / FRAME_RATE
        y_m += float(vy[frame]) / FRAME_RATE

    return states


def trajectory_to_list(states: List[FrameState]) -> List[Dict[str, Any]]:
    """Serialise trajectory to a JSON-compatible list of dicts."""
    return [
        {
            "frame": s.frame,
            "pos_m": list(s.pos_m),
            "pos_px": list(s.pos_px),
            "vel_mps": list(s.vel_mps),
            "on_screen": s.on_screen,
            "spatial_token": s.spatial_token,
            "temporal_step": s.temporal_step,
            "tau": s.tau,
        }
        for s in states
    ]


# ===========================================================================
# 6.  KUBRIC SCENE BUILDER  (called once per video during generation)
# ===========================================================================


def build_scene(
    x0_m: float,
    y0_m: float,
    states: List[FrameState],
    resolution: Tuple[int, int] = (CANVAS_PX, CANVAS_PX),
) -> Tuple:
    """Construct the Kubric scene, renderer, and scratch directory for one video.

    Ball position at each frame is set via direct keyframe insertion, with no
    PyBullet simulation. The orthographic camera covers exactly the
    FIELD_M x FIELD_M visible canvas. Call renderer.render() and save rgba
    frames after this returns; the caller is responsible for
    shutil.rmtree(scratch_dir) on completion.

    Args:
        x0_m: Start x position in world metres.
        y0_m: Start y position in world metres.
        states: The precomputed per-frame trajectory states.
        resolution: Output render resolution (px_x, px_y).

    Returns:
        A tuple (scene, renderer, scratch_dir) for the built scene.

    Raises:
        RuntimeError: If the Kubric Docker environment is not available.
    """
    try:
        import kubric as kb
        from kubric.renderer import Blender
    except ImportError:
        raise RuntimeError("build_scene() requires the Kubric Docker environment.")

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="kubric_nfp_"))

    scene = kb.Scene(
        resolution=resolution,
        frame_start=0,
        frame_end=T - 1,
        frame_rate=FRAME_RATE,
        step_rate=FRAME_RATE,  # no physics sub-steps needed
        gravity=(0, 0, 0),
    )

    renderer = Blender(scene, scratch, use_denoising=False, adaptive_sampling=False)

    # --- Background: flat gray cube acting as floor ---
    gray = kb.FlatMaterial(
        color=kb.Color(0.5, 0.5, 0.5),
        indirect_visibility=False,
    )
    floor = kb.Cube(
        name="floor",
        scale=(KUBRIC_HALF_M, KUBRIC_HALF_M, 0.01),
        position=(0, 0, -0.01),
        material=gray,
        static=True,
        background=True,
    )
    scene.add(floor)

    # --- Ball: flat white sphere ---
    white = kb.FlatMaterial(color=kb.Color(1.0, 1.0, 1.0), indirect_visibility=False)
    ball = kb.Sphere(
        name="ball",
        scale=[BALL_RADIUS_M] * 3,
        position=(x0_m, y0_m, BALL_RADIUS_M),
        material=white,
        static=True,  # positions set via keyframes; no dynamics needed
    )
    scene.add(ball)

    # --- Insert keyframes for every frame ---
    for s in states:
        world_x, world_y = s.pos_m
        ball.position = (world_x, world_y, BALL_RADIUS_M)
        ball.keyframe_insert("position", s.frame)

    # --- Orthographic camera covering exactly FIELD_M x FIELD_M ---
    # OrthographicCamera in Kubric uses orthographic_scale = full width in world units
    scene.camera = kb.OrthographicCamera(
        name="camera",
        position=(0, 0, CAMERA_Z),
        look_at=(0, 0, 0),
        orthographic_scale=FIELD_M,
    )

    return scene, renderer, str(scratch)


# ===========================================================================
# 7.  SANITY CHECKS
# ===========================================================================


def _run_sanity_checks() -> bool:
    """Quick self-test of all tool functions.

    Prints PASS / FAIL per test.

    Returns:
        True if every check passed, False otherwise.
    """
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}")
        if not cond:
            ok = False

    rng = np.random.default_rng(42)

    # --- Coordinate round-trip ---
    for wx, wy in [(-7, 7), (0, 0), (7, -7), (3.5, -2.1)]:
        rx, ry = px_to_world(*world_to_px(wx, wy))
        check(f"coord round-trip ({wx}, {wy})", abs(rx - wx) < 1e-9 and abs(ry - wy) < 1e-9)

    # --- Canvas corners map to pixel corners ---
    check("world (-7,  7) → px (0,   0)", world_to_px(-7, 7) == (0.0, 0.0))
    check("world ( 7, -7) → px (224, 224)", world_to_px(7, -7) == (224.0, 224.0))

    # --- On-screen / off-screen ---
    check("centre on screen", is_on_screen(112, 112))
    check("corner on screen", is_on_screen(0, 0))
    check("edge off screen", not is_on_screen(224, 0))
    check("negative off screen", not is_on_screen(-1, 112))

    # --- Token indexing ---
    check("top-left token = 0", get_spatial_token(0, 0) == 0)
    check("top-right token = 13", get_spatial_token(223, 0) == 13)
    check("second row token = 14", get_spatial_token(0, 16) == 14)
    check("last token = 195", get_spatial_token(223, 223) == 195)
    check("off-screen token = -1", get_spatial_token(224, 0) == -1)

    # --- Temporal step ---
    check("frame 0 → step 0", get_temporal_step(0) == 0)
    check("frame 1 → step 0", get_temporal_step(1) == 0)
    check("frame 2 → step 1", get_temporal_step(2) == 1)
    check("frame 15 → step 7", get_temporal_step(15) == 7)

    # --- Spawn rectangle ---
    check("SPAWN_MIN_M ≤ -9.33", SPAWN_MIN_M <= -9.33)
    check("SPAWN_MAX_M ≥  9.33", SPAWN_MAX_M >= 9.33)
    for _ in range(200):
        x0, y0 = sample_start_position(rng)
        check(
            "spawn in range", SPAWN_MIN_M <= x0 <= SPAWN_MAX_M and SPAWN_MIN_M <= y0 <= SPAWN_MAX_M
        )

    # --- Velocity profiles: cumulative displacement ≤ S_MAX_M per axis ---
    for ptype in FAMILY_A_TYPES:
        for s in [V_MIN_MPS, V_MAX_MPS]:
            speeds = _family_a_speeds(ptype, s)
            cum_disp = np.abs(np.cumsum(speeds / FRAME_RATE)).max()
            check(f"Family A {ptype} s={s}: disp ≤ S_MAX_M", cum_disp <= S_MAX_M + 1e-9)

    for ptype in FAMILY_B_TYPES:
        for delta in [math.radians(45), math.radians(180), math.pi]:
            dirs = _family_b_directions(ptype, 0.0, delta)
            vx = V_MAX_MPS * np.cos(dirs)
            cum_disp_x = np.abs(np.cumsum(vx / FRAME_RATE)).max()
            check(
                f"Family B {ptype} delta={round(math.degrees(delta))}°: x-disp ≤ S_MAX_M",
                cum_disp_x <= S_MAX_M + 1e-9,
            )

    # --- sample_velocity_profile returns correct keys and shapes ---
    for _ in range(50):
        p = sample_velocity_profile(rng)
        check("profile has vx/vy of length T", len(p["vx"]) == T and len(p["vy"]) == T)
        check("profile family is A or B", p["family"] in ("A", "B"))

    # --- Trajectory length and tau keys ---
    x0, y0 = sample_start_position(rng)
    prof = sample_velocity_profile(rng)
    traj = compute_trajectory(x0, y0, prof)
    check("trajectory length = T", len(traj) == T)
    tau_keys = {"speed", "vel_x", "vel_y", "accel_mag", "direction"}
    check("all tau keys present", all(set(s.tau.keys()) == tau_keys for s in traj))
    check("last frame accel_mag = 0", traj[-1].tau["accel_mag"] == 0.0)

    print(f"\n{'All checks passed.' if ok else 'SOME CHECKS FAILED.'}")
    return ok


# ===========================================================================
# 8.  VIDEO GENERATOR  (one video per call)
# ===========================================================================


def profile_from_spec(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a full velocity profile (vx, vy arrays) from a saved parameter entry.

    The entry format is {family, profile_type, speed_mps, direction_deg[, delta_deg]},
    as emitted by dataset_creation/design_decorrelated_stimulus.py (v2 decorrelated
    design). Deterministic: no RNG draws, so the start position (drawn separately)
    remains independent of the profile (S1/S3 preserved).

    Args:
        entry: Saved profile parameter dict.

    Returns:
        A full velocity profile dict with "vx" and "vy" arrays added.
    """
    s = float(entry["speed_mps"])
    theta = math.radians(float(entry["direction_deg"]))
    if entry["family"] == "A":
        speeds = _family_a_speeds(entry["profile_type"], s)
        directions = np.full(T, theta)
    else:
        delta = (
            math.pi
            if entry["profile_type"] == "back_and_forth"
            else math.radians(float(entry["delta_deg"]))
        )
        speeds = np.full(T, s)
        directions = _family_b_directions(entry["profile_type"], theta, delta)
    prof = dict(entry)
    prof["vx"] = speeds * np.cos(directions)
    prof["vy"] = speeds * np.sin(directions)
    return prof


def generate_video(
    video_idx: int,
    output_dir: pathlib.Path,
    rng: np.random.Generator,
    fixed_profile: Optional[Dict[str, Any]] = None,
) -> bool:
    """Generate one video and write it to output_dir / f"v{video_idx:05d}/".

    Skips generation if metadata.json already exists (resume support).

    Args:
        video_idx: Index of the video; sets its output subdirectory name.
        output_dir: Root directory to write the video into.
        rng: NumPy random generator used to draw the start position.
        fixed_profile: If given (v2 decorrelated design), use this profile instead
            of sampling one; the start position is still drawn from rng (S1/S3
            preserved).

    Returns:
        True on success, False if the video was already generated.
    """
    import kubric as kb
    from kubric.renderer import Blender

    out = output_dir / f"v{video_idx:05d}"
    if (out / "metadata.json").exists():
        logger.info("Skipping v%05d (already done)", video_idx)
        return False

    out.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="kubric_nfp_"))

    try:
        # --- S1 and S3: sample start position independently of profile ---
        x0_m, y0_m = sample_start_position(rng)
        profile = fixed_profile if fixed_profile is not None else sample_velocity_profile(rng)
        states = compute_trajectory(x0_m, y0_m, profile)

        # --- Kubric scene ---
        scene = kb.Scene(
            resolution=(CANVAS_PX, CANVAS_PX),
            frame_start=0,
            frame_end=T - 1,
            frame_rate=FRAME_RATE,
            step_rate=FRAME_RATE,
            gravity=(0, 0, 0),
        )
        renderer = Blender(scene, scratch, use_denoising=False, adaptive_sampling=False)

        gray = kb.FlatMaterial(color=kb.Color(0.5, 0.5, 0.5), indirect_visibility=False)
        white = kb.FlatMaterial(color=kb.Color(1.0, 1.0, 1.0), indirect_visibility=False)

        floor = kb.Cube(
            name="floor",
            scale=(KUBRIC_HALF_M, KUBRIC_HALF_M, 0.01),
            position=(0, 0, -0.01),
            material=gray,
            static=True,
            background=True,
        )
        scene.add(floor)

        ball = kb.Sphere(
            name="ball",
            scale=[BALL_RADIUS_M] * 3,
            position=(x0_m, y0_m, BALL_RADIUS_M),
            material=white,
            static=True,
        )
        scene.add(ball)

        # Write a keyframe for every frame using the analytic trajectory
        for s in states:
            ball.position = (s.pos_m[0], s.pos_m[1], BALL_RADIUS_M)
            ball.keyframe_insert("position", s.frame)

        # Orthographic camera: sees exactly FIELD_M × FIELD_M world units
        scene.camera = kb.OrthographicCamera(
            name="camera",
            position=(0, 0, CAMERA_Z),
            look_at=(0, 0, 0),
            orthographic_scale=FIELD_M,
        )

        # --- Render (rgba only - skip depth/segmentation/flow/normals) ---
        data_stack = renderer.render(return_layers=("rgba",))

        for i, frame in enumerate(data_stack["rgba"]):
            kb.write_png(frame, kb.as_path(str(out / f"rgba_{i:05d}.png")))

        # --- Metadata ---
        profile_meta = {k: v for k, v in profile.items() if k not in ("vx", "vy")}

        metadata = {
            "video_id": f"v{video_idx:05d}",
            "start_pos_m": [round(x0_m, 5), round(y0_m, 5)],
            "start_pos_px": list(world_to_px(x0_m, y0_m)),
            "profile": profile_meta,
            "coverage": {
                "spawn_range_m": [SPAWN_MIN_M, SPAWN_MAX_M],
                "s_max_m": round(S_MAX_M, 4),
                "v_max_mps": V_MAX_MPS,
                "field_m": FIELD_M,
            },
            "trajectory": trajectory_to_list(states),
        }
        kb.write_json(metadata, kb.as_path(str(out / "metadata.json")))
        logger.info("Saved v%05d", video_idx)
        return True

    finally:
        shutil.rmtree(str(scratch), ignore_errors=True)


# ===========================================================================
# Entry point
# ===========================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run tool sanity checks and exit (no Kubric needed).",
    )
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--n_videos", type=int, default=3000)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument(
        "--end_idx", type=int, default=-1, help="Inclusive end index. Default -1 = n_videos - 1."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global RNG seed. Each video advances the shared RNG "
        "so shards must use the same seed and non-overlapping ranges.",
    )
    p.add_argument(
        "--profile_spec",
        type=str,
        default=None,
        help="JSON with {'videos': [profile entries]} (v2 decorrelated "
        "design). Video idx uses videos[idx]; positions stay random.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sanity_check:
        print("Running sanity checks...\n")
        _run_sanity_checks()
        raise SystemExit(0)

    if args.output_dir is None:
        raise ValueError("--output_dir is required for generation.")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end_idx = args.end_idx if args.end_idx >= 0 else args.n_videos - 1
    indices = range(args.start_idx, end_idx + 1)

    # Advance the shared RNG to the correct position for this shard.
    # Each video consumes a fixed number of RNG draws; we advance by
    # replaying draws for skipped indices using a fast-forward seed offset.
    # Simple approach: seed = global_seed + start_idx (independent per video).
    logger.info("Generating videos %d–%d → %s", args.start_idx, end_idx, output_dir)

    spec_videos = None
    if args.profile_spec:
        with open(args.profile_spec) as f:
            spec_videos = json.load(f)["videos"]
        logger.info("Using profile spec with %d entries (v2 decorrelated design)", len(spec_videos))

    for idx in indices:
        # Give each video its own sub-RNG derived from global seed + index
        # so shards are reproducible and order-independent.
        video_rng = np.random.default_rng([args.seed, idx])
        fixed = profile_from_spec(spec_videos[idx]) if spec_videos else None
        generate_video(idx, output_dir, video_rng, fixed_profile=fixed)

    logger.info("Done. Generated %d videos in %s", len(indices), output_dir)
