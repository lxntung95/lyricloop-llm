import os

# -------------------------
# Model Configuration
# -------------------------
MODEL_ID = "google/gemma-2b-it"
RANDOM_STATE = 42

# -------------------------
# Path Management
# -------------------------
# Assumes the script is in lyricloop-llm/src/lyricloop/
# Go up 2 levels to reach the lyricloop-llm root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Define standard subfolders
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def ensure_dirs():
    """Initializes the project folder structure if it does not exist."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------
# Global History Template
# -------------------------
def initialize_history():
    """Returns a fresh instance of the experiment history log."""
    return {
        "baseline": {"scores": [], "avg_confidence": [], "samples": {}, "metrics": {}},
        "1.0": {"scores": [], "avg_confidence": [], "samples": {}, "metrics": {}},
        "2.0": {"scores": [], "avg_confidence": [], "samples": {}, "metrics": {}}
    }