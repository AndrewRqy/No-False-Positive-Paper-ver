from dictionary_learning import ActivationBuffer, AutoEncoder, JumpReluAutoEncoder
from dictionary_learning.trainers import *
from dictionary_learning.training import trainSAE
from torch.utils.data import DataLoader
from utils.datasets.activations import ActivationsDataset
import torch
from pathlib import Path
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Train Sparse Autoencoder", add_help=False)
    parser.add_argument("--sae_model", default="jumprelu", type=str)
    parser.add_argument("--activations_dir", required=True, type=str)
    parser.add_argument("--val_activations_dir", required=True, type=str)
    parser.add_argument("--checkpoints_dir", default="./output_dir", type=str)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expansion_factor", type=int, default=1)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save_steps", type=int, default=1_000)
    parser.add_argument("--log_steps", type=int, default=50)
    # JumpRelu
    parser.add_argument("--bandwidth", type=float, default=0.001)
    parser.add_argument("--sparsity_penalty", type=float, default=0.1)
    # Standard
    parser.add_argument("--l1_penalty", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--resample_steps", type=int, default=None)
    parser.add_argument("--dead_feature_threshold", type=int, default=10_000_000)
    parser.add_argument("--dead_penalty_coef", type=float, default=0.0)
    # TopK + Batch TopK
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--auxk_alpha", type=float, default=1/32)
    parser.add_argument("--decay_start", type=int, default=1_000_000)
    # Batch TopK
    parser.add_argument("--threshold_beta", type=float, default=0.999)
    parser.add_argument("--threshold_start_step", type=int, default=1000)
    # MatryoshkaBatchTopK
    parser.add_argument("--group_fractions", type=float, nargs="+")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="VideoMAE SAE")
    parser.add_argument("--no_wandb", default=False, action="store_true")
    # provenance: what the SAE was trained on (recorded in the checkpoint metadata)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--attachment_point", type=str, default=None)

    return parser

class _DeviceDataLoader:
    """Wraps a DataLoader to move batches to the target device on the fly."""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
    def __iter__(self):
        for batch in self.dataloader:
            yield batch.to(self.device)
    def __len__(self):
        return len(self.dataloader)

def train_sae(args):
    dataset = ActivationsDataset(args.activations_dir, device=torch.device("cpu"))
    dataloader = _DeviceDataLoader(
        DataLoader(dataset, batch_size=args.batch_size, shuffle=True), args.device
    )

    val_dataset = ActivationsDataset(args.val_activations_dir, device=torch.device("cpu"))
    val_dataloader = _DeviceDataLoader(
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False), args.device
    )

    sample = next(iter(dataloader))
    print(sample.shape)

    activation_dim = sample.shape[1]
    dictionary_size = args.expansion_factor * activation_dim

    trainers = {
        'jumprelu': JumpReluTrainer,
        'standard': StandardTrainer,
        'batch_top_k': BatchTopKTrainer,
        'top_k': TopKTrainer,
        'matroyshka_batch_top_k': MatroyshkaBatchTopKTrainer,
    }

    # autoencoders = {
    #     'jumprelu': JumpReluAutoEncoder,
    #     'standard': AutoEncoder,
    #     'batch_top_k': BatchTopKSAE,
    #     'top_k': AutoEncoderTopK,
    # }

    trainer_cfg = {
        "trainer": trainers[args.sae_model],
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": args.lr,
        "device": args.device,
        "steps": args.steps,
        "layer": "",
        "lm_name": "",
        "submodule_name": ""
    }

    if args.sae_model == "jumprelu":
        trainer_cfg["bandwidth"] = args.bandwidth
        trainer_cfg["sparsity_penalty"] = args.sparsity_penalty
    if args.sae_model == "standard":
        trainer_cfg["l1_penalty"] = args.l1_penalty
        trainer_cfg["warmup_steps"] = args.warmup_steps
        trainer_cfg["resample_steps"] = args.resample_steps
        trainer_cfg["auxk_alpha"] = args.auxk_alpha
        trainer_cfg["dead_feature_threshold"] = args.dead_feature_threshold
        trainer_cfg["dead_penalty_coef"] = args.dead_penalty_coef
        trainer_cfg["sparsity_warmup_steps"] = None
    if args.sae_model == "top_k" or args.sae_model == "batch_top_k" or args.sae_model == "matroyshka_batch_top_k":
        trainer_cfg["k"] = args.k
        trainer_cfg["auxk_alpha"] = args.auxk_alpha
        trainer_cfg["decay_start"] = args.decay_start
    if args.sae_model == "batch_top_k" or args.sae_model == "matroyshka_batch_top_k":
        trainer_cfg["threshold_beta"] = args.threshold_beta
        trainer_cfg["threshold_start_step"] = args.threshold_start_step
    if args.sae_model == "matroyshka_batch_top_k":
        trainer_cfg["group_fractions"] = args.group_fractions

    dataset_name = Path(args.activations_dir).name
    save_dir = Path(args.checkpoints_dir) / f"{dataset_name}_{args.sae_model}_{args.k}_x{args.expansion_factor}"
    save_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb and args.wandb_entity is not None
    ae = trainSAE(
        data=dataloader,
        val_data=val_dataloader,
        trainer_configs=[trainer_cfg],
        use_wandb=use_wandb,
        wandb_entity=args.wandb_entity or "",
        wandb_project=args.wandb_project,
        steps=args.steps,
        save_steps=[x for x in range(0, args.steps, args.save_steps)],
        save_dir=save_dir,
        log_steps=args.log_steps,
    )

    # Make every saved checkpoint self-describing (review item 5): write a
    # <checkpoint>.meta.json sidecar recording the architecture and the model /
    # layer the SAE was trained on, so utils.sae_io.load_sae can pick the right
    # class and fail clearly on a mismatch.
    from utils.sae_io import save_sae_meta
    for ckpt in save_dir.rglob("*.pt"):
        save_sae_meta(ckpt, sae_type=args.sae_model,
                      activation_dim=activation_dim, dict_size=dictionary_size,
                      expansion_factor=args.expansion_factor,
                      model_name=args.model_name, layer=args.layer,
                      attachment_point=args.attachment_point,
                      k=args.k if args.sae_model in ("top_k", "batch_top_k",
                                                     "matroyshka_batch_top_k") else None)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    train_sae(args)