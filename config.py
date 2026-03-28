"""
Global configuration for the Math Research Paper NLP Pipeline.
All configurable parameters in one place for easy tuning.
"""
import os
import torch

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

# ============================================================
# GPU / DEVICE CONFIGURATION  — Safe & Fast Local GPU
# ============================================================
def get_device():
    """Auto-detect best available device with safety checks."""
    if torch.cuda.is_available():
        # Clear any stale GPU memory
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"[GPU] Using: {gpu_name} ({gpu_mem:.1f} GB)")
        return device
    else:
        print("[CPU] No GPU detected — using CPU (slower)")
        return torch.device("cpu")

DEVICE = get_device()

# GPU memory management
CUDA_MEMORY_FRACTION = 0.85          # Reserve 15% GPU memory for system
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)
    # Enable TF32 for faster computation on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # Auto-tune convolution algorithms

# ============================================================
# MODEL NAMES
# ============================================================
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# ============================================================
# TRAINING HYPERPARAMETERS (GPU-optimized)
# ============================================================
TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 16 if torch.cuda.is_available() else 4,
    "max_length": 512,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": torch.cuda.is_available(),        # Mixed precision on GPU
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 2,
    "pin_memory": torch.cuda.is_available(),
}

# ============================================================
# PIPELINE PARAMETERS
# ============================================================
CHUNK_SIZE = 512          # Tokens per chunk for long documents
CHUNK_OVERLAP = 64        # Overlap between chunks
TOP_K_KEYWORDS = 20       # Number of keywords to extract
TOP_K_SUMMARY_SENTENCES = 5  # Extractive summary length
MIN_SECTION_LENGTH = 50   # Minimum characters for a section

# ============================================================
# GROQ API
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
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
