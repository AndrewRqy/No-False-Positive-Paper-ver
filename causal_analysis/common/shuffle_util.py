"""Shared temporal-shuffle primitives for the restoration experiments (review
item 4).

Both the pair-level (steer_shuffle_restore) and all-class (steer_global_restore)
restoration scripts corrupt motion by permuting the 8 two-frame tubelet blocks
of a clip. Factoring the mechanics here keeps them from diverging. NOTE the two
scripts differ deliberately in their sampling POLICY, documented at each call
site:
  - steer_shuffle_restore: one permutation per class-side evaluation set, applied
    to every clip on that side and shared across conditions;
  - steer_global_restore: one permutation per video.
"""


def block_order(perm):
    """8 block indices -> the 16 frame indices that realize that block order."""
    order = []
    for p in perm:
        order += [2 * p, 2 * p + 1]
    return order


def shuffle_blocks(pv, perm):
    """pv [...,16,...] indexed on the frame axis (dim 1); perm is a length-8 block
    permutation applied to every row."""
    return pv[:, block_order(perm)]


def random_block_perm(rng, n_blocks=8):
    """A non-identity permutation of the tubelet blocks, drawn from `rng`
    (numpy RandomState / Generator)."""
    perm = list(rng.permutation(n_blocks))
    while perm == list(range(n_blocks)):
        perm = list(rng.permutation(n_blocks))
    return perm
