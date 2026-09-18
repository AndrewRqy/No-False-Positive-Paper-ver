from transformers import AutoImageProcessor, Dinov2Model
import torch

class Dino:
    def __init__(self, model_name="dinov2-base", device=torch.device("cuda")):
        self.device = device
        self.model = Dinov2Model.from_pretrained(f"facebook/{model_name}").to(device)
        self.processor = AutoImageProcessor.from_pretrained(f"facebook/{model_name}")
        self.register = {}
        self._attached = False
        self._sae = None
        self._attach_key = None

    def attach(self, attachment_point, layer, sae=None):
        if attachment_point != "pooler_output":
            raise NotImplementedError(f"Attachment point '{attachment_point}' not implemented for Dino")
        self._attached = True
        self._attach_key = f"{attachment_point}_{layer}"
        self._sae = sae
        self.register[self._attach_key] = []

    def encode(self, inputs):
        if self._attached:
            for key in self.register:
                self.register[key] = []
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        acts = outputs.pooler_output  # [B, 768]
        if self._attached:
            enc = self._sae.encode(acts) if self._sae is not None else acts
            self.register[self._attach_key].append(enc.detach().cpu())
        return acts
