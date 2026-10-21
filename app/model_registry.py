"""Generation choices and their shared download packages."""
import sys

MODELS = {
    'bf16': {'label': 'YuE2 BF16', 'description': 'Full precision · Metal / MLX CUDA', 'backend': 'mlx'},
    'mps-pytorch': {'label': 'Existing YuE2 MPS', 'description': 'Reuse an installed official PyTorch BF16 model', 'backend': 'mps-pytorch'},
    'cuda-bf16': {'label': 'YuE2 CUDA BF16', 'description': 'Original YuE2 · full precision', 'backend': 'cuda'},
    'cuda-fp8': {'label': 'YuE2 CUDA FP8', 'description': 'Experimental FP8 token generation · BF16 synthesis', 'backend': 'cuda'},
}


def default_model():
    return 'cuda-bf16' if sys.platform == 'win32' else 'bf16'


def is_cuda(model):
    return MODELS.get(model, {}).get('backend') == 'cuda'


def is_external_mps(model):
    return MODELS.get(model, {}).get('backend') == 'mps-pytorch'


def is_upstream_torch(model):
    return is_cuda(model) or is_external_mps(model)


def available_models():
    if sys.platform == 'win32':
        return {'cuda-bf16': MODELS['cuda-bf16']}
    return {key:value for key,value in MODELS.items() if sys.platform != 'darwin' or not is_cuda(key)}
