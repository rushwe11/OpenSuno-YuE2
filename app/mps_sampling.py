"""Numerically stable sampling shim for the official YuE2 runtime on Apple MPS."""

from functools import wraps
import torch


def mps_available():
    backend = getattr(getattr(torch, 'backends', None), 'mps', None)
    return bool(backend and backend.is_available())


def safe_mps_logits(logits):
    """Move MPS sampling arithmetic to FP32 CPU and repair non-finite logits."""
    return torch.nan_to_num(logits.float().cpu(), nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)


def install_mps_safe_sampling():
    """Upstream Direct mode keeps historical BF16 sampling, which can produce NaNs on MPS."""
    from yue2 import sampling as upstream
    if getattr(upstream.distribution, '_opensuno_mps_safe', False):
        return
    original = upstream.distribution

    @wraps(original)
    def distribution(logits, *args, **kwargs):
        device = getattr(logits, 'device', None)
        if device is not None and getattr(device, 'type', None) == 'mps':
            logits = safe_mps_logits(logits)
        return original(logits, *args, **kwargs)
    distribution._opensuno_mps_safe = True
    upstream.distribution = distribution
