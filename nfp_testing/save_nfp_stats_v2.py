"""
Compute and save NFP statistics on the v2 dataset for the main SAE, in the same format
as sae_nfp.pt ({t_stat, p_val} [6144, 5]), so every steering script can take
--nfp_results sae_nfp_v2.pt and run on the v2 flag set (109 features).

Usage (from sae-for-vlm/):
  python analysis/save_nfp_stats_v2.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder


def main():
    d = torch.load("local_runs/nfp_results/ball_raw_acts_v2.pt", map_location="cpu")
    ball, tau, mask = d["ball"].float(), d["tau"].float(), d["mask"]
    sae = AutoEncoder.from_pretrained("local_runs/sae/ae.pt", device="cuda:0")
    sae.eval()
    N, T, _ = ball.shape
    with torch.no_grad():
        feats = sae.encode(ball.reshape(N * T, -1).to("cuda:0")).cpu().reshape(N, T, -1)
    feats = feats * mask.unsqueeze(-1).float()
    psi_c = feats - feats.mean(1, keepdim=True)
    tau_c = tau - tau.mean(1, keepdim=True)
    C = torch.einsum("btd,btk->bdk", psi_c, tau_c).numpy() / T
    D = C.shape[1]
    t = np.zeros((D, 5), np.float32); p = np.ones_like(t)
    for k in range(5):
        t[:, k], p[:, k] = stats.ttest_1samp(C[:, :, k], 0.0)
    sig = ((p < 0.05 / D).any(1)).sum()
    print(f"v2 flags: {sig} features")
    torch.save({"t_stat": torch.from_numpy(t), "p_val": torch.from_numpy(p),
                "tau": tau, "tau_keys": ["speed", "vel_x", "vel_y", "accel_mag", "direction"]},
               "local_runs/nfp_results/sae_nfp_v2.pt")
    print("saved -> local_runs/nfp_results/sae_nfp_v2.pt")


if __name__ == "__main__":
    main()
