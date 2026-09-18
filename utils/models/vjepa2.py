"""V-JEPA2 wrapper (strong second model for the cross-model NFP study).

facebook/vjepa2-vitl-fpc16-256-ssv2: 16 input frames at 256, tubelet 2 ->
8 temporal steps, 16x16 patch grid (256 spatial cells), 24 layers, 1024-d,
174-way SSv2 attentive probe.

Token layout: FRAME-MAJOR with no CLS token:
    index = t * 256 + s        (t = tubelet 0..7, s = spatial cell 0..255)
The processor emits `pixel_values_videos` [B, 16, 3, 256, 256].
"""
from transformers import VJEPA2Model, AutoVideoProcessor
import torch
import torch.nn as nn
from typing import Tuple, Union


class VJEPA2:
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = VJEPA2Model.from_pretrained(model_name).to(device)
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.register = {}
        self.attach_methods = {
            "post_mlp_residual": self._attach_post_mlp_residual,
            "pooler_post_selfattn": self._attach_pooler_post_selfattn,
        }

    def encode(self, inputs):
        for key in self.register:
            self.register[key] = []
        pv = inputs["pixel_values_videos"].to(self.device)
        with torch.no_grad():
            self.model(pixel_values_videos=pv)

    def attach(self, attachment_point, layer, sae=None):
        if attachment_point not in self.attach_methods:
            raise NotImplementedError(
                f"Attachment point '{attachment_point}' not implemented for VJEPA2")
        self.attach_methods[attachment_point](layer, sae)
        self.register[f"{attachment_point}_{layer}"] = []

    def _attach_post_mlp_residual(self, layer, sae):
        self.model.encoder.layer[layer] = _LayerPostMlpResidual(
            self.model.encoder.layer[layer], sae, layer, self.register)

    def _attach_pooler_post_selfattn(self, layer, sae):
        # token-structured stream inside the SSv2 attentive probe: output of
        # pooler self-attention layer `layer`, before the query cross-attention
        if sae is not None:
            raise NotImplementedError("SAE attachment not supported at the pooler")
        from transformers import VJEPA2ForVideoClassification
        self.model = VJEPA2ForVideoClassification.from_pretrained(
            self.model_name).to(self.device)
        key = f"pooler_post_selfattn_{layer}"

        def hook(_mod, _inp, out):
            acts = out[0] if isinstance(out, tuple) else out
            self.register[key].append(acts.detach().cpu())

        self.model.pooler.self_attention_layers[layer].register_forward_hook(hook)


class _LayerPostMlpResidual(nn.Module):
    def __init__(self, base_layer, sae, layer_idx, register):
        super().__init__()
        self.base_layer = base_layer
        self.sae = sae
        self.layer_idx = layer_idx
        self.register = register

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs
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
