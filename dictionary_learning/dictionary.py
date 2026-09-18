"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init
import einops


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(
                f_ghost
            )  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.bias.data *= scale

    def normalize_decoder(self):
        norms = t.norm(self.decoder.weight, dim=0).to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)

        if t.allclose(norms, t.ones_like(norms)):
            return
        print("Normalizing decoder weights")

        test_input = t.randn(10, self.activation_dim)
        initial_output = self(test_input)

        # dead features can have exactly-zero decoder columns; leave them at zero
        safe = t.where(norms > 1e-8, norms, t.ones_like(norms))
        self.decoder.weight.data /= safe

        new_norms = t.norm(self.decoder.weight, dim=0)
        live = norms > 1e-8
        assert t.allclose(new_norms[live], t.ones_like(new_norms[live]))

        self.encoder.weight.data *= safe[:, None]
        self.encoder.bias.data *= safe

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert t.allclose(initial_output, new_output, atol=1e-4)


    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        # If training the SAE using the April update, the decoder weights are not normalized
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class LinearDict(Dictionary, nn.Module):
    """
    A linear decomposition (PCA or ICA) wrapped in the Dictionary interface so it
    can be swapped in for an SAE in the activation-extraction / monosemanticity
    pipeline.

    A fitted decomposition is stored as three tensors:
        mean   : (activation_dim,)            data mean removed before projecting
        E      : (n_components, activation_dim)  encode  matrix:  s = (x - mean) @ E.T
        D      : (n_components, activation_dim)  decode  matrix:  x_hat = s @ D + mean
    For PCA, E = D = components_. For ICA, E = components_ (unmixing) and
    D = mixing_.T, so encode/decode are exact inverses of sklearn's
    transform / inverse_transform.

    Because PCA/ICA components are *signed* while the monosemanticity metric assumes
    non-negative "feature present" activations, encode() supports three modes:
        "sign_split" (default, primary): each component c -> [ReLU(c), ReLU(-c)],
            doubling the feature count to 2*n_components. Each half-feature has the
            same "concept present, >=0" semantics as a ReLU SAE feature, and removes
            ICA's arbitrary sign. decode recovers c = c_plus - c_minus exactly.
        "abs": magnitude |c| (n_components features). Robustness variant; merges the
            two poles of a component into one feature.
        "signed": raw signed projection (n_components features). Not recommended for
            the MS metric; provided for completeness.
    """

    def __init__(self, activation_dim, n_components, mode="sign_split"):
        super().__init__()
        assert mode in ("sign_split", "abs", "signed")
        self.activation_dim = activation_dim
        self.n_components = n_components
        self.mode = mode
        # If the fit was done on L2-normalized activations, encode must apply the
        # SAME normalization or the projection space is inconsistent (review item 6).
        # Loaded from the checkpoint's `l2_normalize` flag by from_pretrained.
        self.l2_normalize = False
        self.dict_size = 2 * n_components if mode == "sign_split" else n_components
        self.register_buffer("mean", t.zeros(activation_dim))
        self.register_buffer("E", t.zeros(n_components, activation_dim))
        self.register_buffer("D", t.zeros(n_components, activation_dim))

    def encode(self, x):
        if self.l2_normalize:
            # match the preprocessing applied when the components were fit
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        s = (x - self.mean) @ self.E.t()  # (..., n_components), signed
        if self.mode == "sign_split":
            return t.cat([t.relu(s), t.relu(-s)], dim=-1)
        elif self.mode == "abs":
            return s.abs()
        else:
            return s

    def decode(self, f):
        if self.mode == "sign_split":
            s = f[..., : self.n_components] - f[..., self.n_components:]
        elif self.mode == "abs":
            # Sign is lost under |c|; reconstruct from magnitude (decode output is
            # only used to continue the forward pass, never for the MS metric).
            s = f
        else:
            s = f
        return s @ self.D + self.mean

    def forward(self, x, output_features=False, ghost_mask=None):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        return x_hat

    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, mode="sign_split"):
        """
        Load a fitted decomposition saved by analysis/fit_pca_ica.py.

        The checkpoint is a dict with keys: mean, E, D, method, activation_dim,
        n_components. `mode` selects how signed components are mapped to features
        (overridable at eval time without refitting).
        """
        ckpt = t.load(path, map_location="cpu")
        activation_dim = int(ckpt["activation_dim"])
        n_components = int(ckpt["n_components"])
        obj = cls(activation_dim, n_components, mode=mode)
        # NOTE: with l2_normalize=True the components live in the unit-sphere space,
        # so decode() reconstructs the NORMALIZED activation, not the original norm.
        obj.l2_normalize = bool(ckpt.get("l2_normalize", False))
        obj.mean.copy_(ckpt["mean"].to(dtype))
        obj.E.copy_(ckpt["E"].to(dtype))
        obj.D.copy_(ckpt["D"].to(dtype))
        if device is not None:
            obj.to(dtype=dtype, device=device)
        return obj


class PCADict(LinearDict):
    """PCA decomposition. Thin alias of LinearDict for an explicit --sae_model name."""
    method = "pca"


class ICADict(LinearDict):
    """ICA decomposition. Thin alias of LinearDict for an explicit --sae_model name."""
    method = "ica"


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x

    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None):
        """
        Load a pretrained dictionary from a file.
        """
        return cls(None)


class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """

    def __init__(self, activation_dim, dict_size, initialization="default", device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(t.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(t.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == "default":
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)

    def encode(self, x: t.Tensor, return_gate:bool=False, normalize_decoder:bool=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag

        if normalize_decoder:
            # If the SAE is trained without ConstrainedAdam, the decoder vectors are not normalized
            # Normalizing after encode, and renormalizing before decode to enable comparability
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f: t.Tensor, normalize_decoder:bool=False):
        if normalize_decoder:
            # If the SAE is trained without ConstrainedAdam, the decoder vectors are not normalized
            # Normalizing after encode, and renormalizing before decode to enable comparability
            f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(self, x:t.Tensor, output_features:bool=False, normalize_decoder:bool=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        if normalize_decoder:
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.decoder_bias.data *= scale
        self.mag_bias.data *= scale
        self.gate_bias.data *= scale

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class JumpReluAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, device="cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(t.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(
            t.nn.init.kaiming_uniform_(t.empty(dict_size, activation_dim, device=device))
        )
        self.b_dec = nn.Parameter(t.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(t.ones(dict_size, device=device) * 0.001)  # Appendix I

        self.apply_b_dec_to_input = False

        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x, output_pre_jump=False):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))

        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        self.b_enc.data *= scale
        self.threshold.data *= scale

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        load_from_sae_lens: bool = False,
        dtype: t.dtype = t.float32,
        device: t.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        if not load_from_sae_lens:
            state_dict = t.load(path)
            activation_dim, dict_size = state_dict["W_enc"].shape
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size)
            autoencoder.load_state_dict(state_dict)
            autoencoder = autoencoder.to(dtype=dtype, device=device)
        else:
            from sae_lens import SAE

            sae, cfg_dict, _ = SAE.from_pretrained(**kwargs)
            assert (
                cfg_dict["finetuning_scaling_factor"] == False
            ), "Finetuning scaling factor not supported"
            dict_size, activation_dim = cfg_dict["d_sae"], cfg_dict["d_in"]
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size, device=device)
            autoencoder.load_state_dict(sae.state_dict())
            autoencoder.apply_b_dec_to_input = cfg_dict["apply_b_dec_to_input"]

        if device is not None:
            device = autoencoder.W_enc.device
        return autoencoder.to(dtype=dtype, device=device)


# TODO merge this with AutoEncoder
class AutoEncoderNew(Dictionary, nn.Module):
    """
    The autoencoder architecture and initialization used in https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        """
        if not output_features:
            return self.decode(self.encode(x))
        else:  # TODO rewrite so that x_hat depends on f
            f = self.encode(x)
            x_hat = self.decode(f)
            # multiply f by decoder column norms
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
            return x_hat, f

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderNew(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
