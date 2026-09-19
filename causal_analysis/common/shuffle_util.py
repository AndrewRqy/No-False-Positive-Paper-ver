"""Temporal-Shuffle Primitives - shared block-permutation helpers for restoration experiments.

Factored out for the restoration experiments (review item 4).

Both the pair-level (steer_shuffle_restore) and all-class (steer_global_restore)
restoration scripts corrupt motion by permuting the 8 two-frame tubelet blocks
of a clip. Factoring the mechanics here keeps them from diverging. NOTE the two
scripts differ deliberately in their sampling POLICY, documented at each call
site:
  - steer_shuffle_restore: one permutation per class-side evaluation set, applied
    to every clip on that side and shared across conditions;
  - steer_global_restore: one permutation per video.
"""

from typing import Any, List


def block_order(perm: List[int]) -> List[int]:
    """Expand an 8-block permutation into the 16 frame indices that realize it.

    Args:
        perm: A permutation of the 8 two-frame tubelet block indices.

    Returns:
        The 16 frame indices (two consecutive frames per block) in shuffled order.
    """
    order = []
    for p in perm:
        order += [2 * p, 2 * p + 1]
    return order


def shuffle_blocks(pv: Any, perm: List[int]) -> Any:
    """Reorder a clip's frames by a block permutation.

    Args:
        pv: A tensor indexed on the frame axis (dim 1), shape [..., 16, ...].
        perm: A length-8 block permutation applied to every row.

    Returns:
        The input reindexed along the frame axis into the permuted block order.
    """
    return pv[:, block_order(perm)]


def random_block_perm(rng: Any, n_blocks: int = 8) -> List[int]:
    """Draw a non-identity permutation of the tubelet blocks.

    Args:
        rng: A numpy RandomState or Generator used to draw the permutation.
        n_blocks: Number of tubelet blocks to permute.

    Returns:
        A block permutation guaranteed not to equal the identity ordering.
    """
    perm = list(rng.permutation(n_blocks))
    while perm == list(range(n_blocks)):
        perm = list(rng.permutation(n_blocks))
    return perm
