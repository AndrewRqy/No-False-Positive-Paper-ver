from .dictionary import AutoEncoder, GatedAutoEncoder, JumpReluAutoEncoder, LinearDict, PCADict, ICADict, IdentityDict
try:
    # ActivationBuffer pulls in nnsight, which is only needed for SAE *training*.
    # Guard it so activation extraction / PCA-ICA fitting work without nnsight installed.
    from .buffer import ActivationBuffer
except ImportError:
    ActivationBuffer = None