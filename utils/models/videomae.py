from transformers import VideoMAEModel, VideoMAEImageProcessor
import torch
import torch.nn as nn
from typing import Tuple, Union


class VideoMAE:
    def __init__(self, model_name, device):
        self.device = device
        self.model = VideoMAEModel.from_pretrained(model_name).to(device)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
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
            raise NotImplementedError(f"Attachment point '{attachment_point}' not implemented for VideoMAE")
        self.attach_methods[attachment_point](layer, sae)
        self.register[f"{attachment_point}_{layer}"] = []

    def _attach_post_mlp_residual(self, layer, sae):
        self.model.encoder.layer[layer] = VideoMAELayerPostMlpResidual(
            self.model.encoder.layer[layer],
            sae,
            layer,
            self.register,
        )


class VideoMAELayerPostMlpResidual(nn.Module):
    def __init__(self, base_layer, sae, layer_idx, register):
        super().__init__()
        self.base_layer = base_layer
        self.sae = sae
        self.layer_idx = layer_idx
        self.register = register

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # Some transformers versions pass head_mask positionally
        # (layer_module(hidden_states, layer_head_mask)), others as a kwarg.
        # Forward whatever we got to the base layer; head masking is unused here.
        kwargs.pop('head_mask', None)
        outputs = self.base_layer(hidden_states, *args, **kwargs)

        is_tuple = not isinstance(outputs, torch.Tensor)
        if is_tuple:
            acts = outputs[0]
            rest = outputs[1:]
        else:
            acts = outputs
            rest = ()

        if self.sae is not None:
            acts = self.sae.encode(acts)
            self.register[f"post_mlp_residual_{self.layer_idx}"].append(acts.detach().cpu())
            acts = self.sae.decode(acts)
        else:
            self.register[f"post_mlp_residual_{self.layer_idx}"].append(acts.detach().cpu())

        # preserve the base layer's return type: newer transformers vision
        # layers return a bare tensor that flows into the final layernorm
        return ((acts,) + rest) if is_tuple else acts
