"""TimeSformer wrapper (weak-model control for the cross-model NFP study).

facebook/timesformer-base-finetuned-ssv2: 8 input frames at 224, 14x14 patch
grid, 12 layers, 768-d, divided space-time attention, 174-way SSv2 head.

Token layout note (verified against HF TimesformerEmbeddings.forward): after
the time-embedding block the token stream is PATCH-MAJOR:
    index = 1 + s * T + t      (CLS at 0, s = spatial cell 0..195, t = frame 0..7)
unlike VideoMAE's frame-major t * 196 + s (no CLS).
"""

from transformers import TimesformerModel, AutoImageProcessor
import torch
import torch.nn as nn
from typing import Tuple, Union


class Timesformer:
    def __init__(self, model_name, device):
        self.device = device
        self.model = TimesformerModel.from_pretrained(model_name).to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.num_frames = self.model.config.num_frames  # 8
        self.register = {}
        self.attach_methods = {
            "post_mlp_residual": self._attach_post_mlp_residual,
        }

    def encode(self, inputs):
        for key in self.register:
            self.register[key] = []
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            self.model(pixel_values=pixel_values)

    def attach(self, attachment_point, layer, sae=None):
        if attachment_point not in self.attach_methods:
            raise NotImplementedError(
                f"Attachment point '{attachment_point}' not implemented for Timesformer"
            )
        self.attach_methods[attachment_point](layer, sae)
        self.register[f"{attachment_point}_{layer}"] = []

    def _attach_post_mlp_residual(self, layer, sae):
        self.model.encoder.layer[layer] = _LayerPostMlpResidual(
            self.model.encoder.layer[layer], sae, layer, self.register
        )


class _LayerPostMlpResidual(nn.Module):
    def __init__(self, base_layer, sae, layer_idx, register):
        super().__init__()
        self.base_layer = base_layer
        self.sae = sae
        self.layer_idx = layer_idx
        self.register = register

    def forward(
        self, hidden_states: torch.Tensor, *args, **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        kwargs.pop("head_mask", None)
        outputs = self.base_layer(hidden_states, *args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            acts, rest = outputs, ()
        else:
            acts, rest = outputs[0], outputs[1:]
        if self.sae is not None:
            acts = self.sae.encode(acts)
            self.register[f"post_mlp_residual_{self.layer_idx}"].append(acts.detach().cpu())
            acts = self.sae.decode(acts)
        else:
            self.register[f"post_mlp_residual_{self.layer_idx}"].append(acts.detach().cpu())
        return (acts,) + rest
