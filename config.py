"""
Global configuration for the Math Research Paper NLP Pipeline.
All configurable parameters in one place for easy tuning.
"""
import os
import torch
from runtime_settings import get_groq_api_key

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create directories
for d in [CACHE_DIR, RESULTS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)


def has_mps() -> bool:
    """Return True when Apple Metal Performance Shaders is available."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available() and mps_backend.is_built())


def clear_device_cache(device: torch.device | None = None) -> None:
    """Release cached accelerator memory when supported."""
    device_type = device.type if device is not None else (
        "cuda" if torch.cuda.is_available() else "mps" if has_mps() else "cpu"
    )

    try:
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def get_device_name(device: torch.device) -> str:
    """Return a human-friendly device name."""
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


def get_accelerator_memory_gb(device: torch.device | None = None) -> float:
    """Return total accelerator memory when it can be queried safely."""
    active_device = device or DEVICE
    if active_device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================
def get_device():
    """Auto-detect the best available execution device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        clear_device_cache(device)
        gpu_name = get_device_name(device)
        gpu_mem = get_accelerator_memory_gb(device)
        print(f"[GPU] Using: {gpu_name} ({gpu_mem:.1f} GB)")
        return device

    if has_mps():
        device = torch.device("mps")
        print(f"[GPU] Using: {get_device_name(device)}")
        return device

    if torch.version.cuda is None:
        print("[CPU] No accelerator is available to PyTorch in this environment.")
    else:
        print(f"[CPU] CUDA build detected (torch CUDA {torch.version.cuda}) but no usable GPU was found by PyTorch.")
    return torch.device("cpu")


DEVICE = get_device()
DEVICE_TYPE = DEVICE.type
DEVICE_NAME = get_device_name(DEVICE)
ACCELERATOR_AVAILABLE = DEVICE_TYPE != "cpu"
GPU_TOTAL_MEMORY_GB = get_accelerator_memory_gb(DEVICE)
LOW_MEMORY_THRESHOLD_GB = 6.0
LOW_MEMORY_MODE = (
    (DEVICE_TYPE == "cuda" and 0.0 < GPU_TOTAL_MEMORY_GB < LOW_MEMORY_THRESHOLD_GB)
    or DEVICE_TYPE == "mps"
)
LOW_VRAM_MODE = LOW_MEMORY_MODE
TRANSFORMER_TORCH_DTYPE = torch.float16 if DEVICE_TYPE == "cuda" else torch.float32
ZERO_SHOT_PIPELINE_DEVICE = 0 if DEVICE_TYPE == "cuda" else -1
EMBED_BATCH_SIZE = 8 if LOW_MEMORY_MODE else (12 if DEVICE_TYPE == "cpu" else 16)
SEARCH_BATCH_SIZE = 16 if LOW_MEMORY_MODE else (24 if DEVICE_TYPE == "cpu" else 32)

if LOW_MEMORY_MODE:
    if DEVICE_TYPE == "cuda" and GPU_TOTAL_MEMORY_GB > 0:
        print(f"[Runtime] Low-memory mode enabled ({GPU_TOTAL_MEMORY_GB:.1f} GB < {LOW_MEMORY_THRESHOLD_GB:.1f} GB).")
    else:
        print(f"[Runtime] Low-memory mode enabled for {DEVICE_NAME}.")

# GPU memory management
CUDA_MEMORY_FRACTION = 0.85
if DEVICE_TYPE == "cuda" and torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ============================================================
# MODEL NAMES
# ============================================================
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 16 if DEVICE_TYPE == "cuda" else (8 if ACCELERATOR_AVAILABLE else 4),
    "max_length": 512,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": DEVICE_TYPE == "cuda",
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 2,
    "pin_memory": DEVICE_TYPE == "cuda",
}

# ============================================================
# PIPELINE PARAMETERS
# ============================================================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_KEYWORDS = 20
TOP_K_SUMMARY_SENTENCES = 5
MIN_SECTION_LENGTH = 50

# ============================================================
# GROQ API
# ============================================================
GROQ_API_KEY = get_groq_api_key()
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "mixtral-8x7b-32768"
GROQ_MAX_TOKENS = 4096
GROQ_TEMPERATURE = 0.3

# ============================================================
# SECTION LABELS for classification
# ============================================================
SECTION_LABELS = [
    "introduction",
    "related_work",
    "methodology",
    "theory",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "abstract",
    "references",
]

# Math equation type labels
EQUATION_TYPES = [
    "algebra",
    "calculus",
    "optimization",
    "probability",
    "linear_algebra",
    "differential_equations",
    "number_theory",
    "statistics",
    "geometry",
    "other",
]
