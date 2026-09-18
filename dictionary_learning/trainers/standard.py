"""
Implements the standard SAE training scheme.
"""
import torch as t
from typing import Optional

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import AutoEncoder
from collections import namedtuple

class StandardTrainer(SAETrainer):
    """
    Standard SAE training scheme following Towards Monosemanticity. Decoder column norms are constrained to 1.
    Optionally adds an auxiliary reconstruction loss (auxk_alpha > 0) to revive dead neurons instead of
    (or in addition to) periodic resampling.
    """
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=AutoEncoder,
                 lr:float=1e-3,
                 l1_penalty:float=1e-1,
                 warmup_steps:int=1000, # lr warmup period at start of training and after each resample
                 sparsity_warmup_steps:Optional[int]=2000, # sparsity warmup period at start of training
                 decay_start:Optional[int]=None, # decay learning rate after this many steps
                 resample_steps:Optional[int]=None, # how often to resample neurons
                 auxk_alpha:float=0.0, # weight of auxiliary loss; 0 disables it
                 dead_feature_threshold:int=10_000_000, # tokens of inactivity before a feature is "dead"
                 dead_penalty_coef:float=0.0, # weight of direct pre-activation penalty for dead features; 0 disables it
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='StandardTrainer',
                 submodule_name:Optional[str]=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None

        # auxiliary loss state
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = dead_feature_threshold
        self.dead_penalty_coef = dead_penalty_coef
        self.top_k_aux = activation_dim // 2  # heuristic: up to d/2 dead features contribute
        if self.auxk_alpha > 0:
            self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=self.device)

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # get norm of the living neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            # resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads] = 0.


            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            ## encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            ## decoder weight
            state_dict[3]['exp_avg'][:,deads] = 0.
            state_dict[3]['exp_avg_sq'][:,deads] = 0.
    
    def get_auxiliary_loss(self, residual: t.Tensor, pre_acts: t.Tensor) -> t.Tensor:
        """
        Auxiliary reconstruction loss for dead features.

        For each dead feature (inactive for >= dead_feature_threshold tokens), we treat
        its pre-ReLU activation as a proxy for how useful it would be. We select the
        top-k_aux dead features by pre-activation, ReLU them, and compute how well they
        would reconstruct the residual. This gives dead features a gradient signal
        proportional to their potential utility, reviving the most useful ones first.

        Loss is normalized by the residual variance so its scale stays stable as the main
        reconstruction improves over training.
        """
        dead = self.num_tokens_since_fired >= self.dead_feature_threshold
        if dead.sum() == 0:
            return t.tensor(0.0, dtype=residual.dtype, device=residual.device)

        k_aux = min(self.top_k_aux, int(dead.sum()))

        # mask alive features to -inf so only dead ones can be selected
        dead_pre = t.where(dead[None], pre_acts, t.full_like(pre_acts, float('-inf')))
        topk_vals, topk_idx = dead_pre.topk(k_aux, sorted=False)

        # reconstruct residual using dead features only (decoder weights, no bias)
        aux_acts = t.zeros_like(pre_acts).scatter_(-1, topk_idx, t.relu(topk_vals))
        x_aux = self.ae.decoder(aux_acts)

        l2_aux = (residual.float() - x_aux.float()).pow(2).sum(dim=-1).mean()

        # normalize by variance of residual to keep aux loss scale-invariant
        residual_mean = residual.mean(dim=0, keepdim=True)
        denom = (residual.float() - residual_mean.float()).pow(2).sum(dim=-1).mean()
        return (l2_aux / denom).nan_to_num(0.0)

    def loss(self, x, step: int, logging=False, **kwargs):

        sparsity_scale = self.sparsity_warmup_fn(step)

        # expose pre-ReLU activations (needed for aux loss and identical to encode() otherwise)
        pre_acts = self.ae.encoder(x - self.ae.bias)
        f = t.relu(pre_acts)
        x_hat = self.ae.decode(f)

        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        # update firing trackers
        did_fire = (f > 0).any(dim=0)
        if self.steps_since_active is not None:
            self.steps_since_active[~did_fire] += 1
            self.steps_since_active[did_fire] = 0
        if self.auxk_alpha > 0:
            self.num_tokens_since_fired += x.size(0)
            self.num_tokens_since_fired[did_fire] = 0

        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        aux_loss_val = 0.0
        if self.auxk_alpha > 0:
            aux_loss = self.get_auxiliary_loss((x - x_hat).detach(), pre_acts.detach())
            loss = loss + self.auxk_alpha * aux_loss
            aux_loss_val = aux_loss.item()

        dead_pen_val = 0.0
        if self.dead_penalty_coef > 0:
            # penalize features whose mean pre-activation across the batch is negative;
            # gradient flows to encoder weights/bias, pushing dead features back to life
            dead_pen = t.relu(-pre_acts.mean(dim=0)).mean()
            loss = loss + self.dead_penalty_coef * dead_pen
            dead_pen_val = dead_pen.item()

        if not logging:
            return loss
        else:
            n_dead = int((self.num_tokens_since_fired >= self.dead_feature_threshold).sum()) \
                     if self.auxk_alpha > 0 else -1
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss':        l2_loss.item(),
                    'mse_loss':       recon_loss.item(),
                    'sparsity_loss':  l1_loss.item(),
                    'aux_loss':       aux_loss_val,
                    'dead_pen':       dead_pen_val,
                    'dead_features':  n_dead,
                    'loss':           loss.item(),
                }
            )


    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'StandardTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'resample_steps' : self.resample_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'auxk_alpha' : self.auxk_alpha,
            'dead_feature_threshold' : self.dead_feature_threshold,
            'dead_penalty_coef' : self.dead_penalty_coef,
            'steps' : self.steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }


class StandardTrainerAprilUpdate(SAETrainer):
    """
    Standard SAE training scheme following the Anthropic April update. Decoder column norms are NOT constrained to 1.
    This trainer does not support resampling or ghost gradients. This trainer will have fewer dead neurons than the standard trainer.
    """
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=AutoEncoder,
                 lr:float=1e-3,
                 l1_penalty:float=1e-1,
                 warmup_steps:int=1000, # lr warmup period at start of training
                 sparsity_warmup_steps:Optional[int]=2000, # sparsity warmup period at start of training
                 decay_start:Optional[int]=None, # decay learning rate after this many steps
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='StandardTrainerAprilUpdate',
                 submodule_name:Optional[str]=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None, sparsity_warmup_steps=sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def loss(self, x, step: int, logging=False, **kwargs):

        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = (f * self.ae.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()

        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'sparsity_loss' : l1_loss.item(),
                    'loss' : loss.item()
                }
            )


    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'StandardTrainerAprilUpdate',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'steps' : self.steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }

