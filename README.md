NAME

    LyricLoop LLM

---

PROJECT OBJECTIVE

    LyricLoop bridges the gap between semantic LLM text and professional musical phrasing. This framework fine-tunes Google's Gemma-2b-it to generate lyrics adhering to specific structures (Verse, Chorus, Bridge) and genre-specific stylings (Electronic, Pop, Rock, Hip-Hop).

---

LANGUAGE / STACK

    Python | PyTorch, Hugging Face (Transformers, PEFT, TRL), Streamlit

---

TECHNICAL METHODOLOGY

    - Fine-Tuning: Implemented Low-Rank Adaptation (LoRA) to specialize the model in rhythmic patterns while preserving base reasoning.
    - Optimization: Used 4-bit Quantization (QLoRA) via bitsandbytes to reduce the memory footprint during training.
    - Instruction Tuning: Supervised Fine-Tuning (SFT) with custom templates to enforce structural and genre constraints.

---

PROJECT STRUCTURE

    - app.py: main streamlit application entry point and UI logic.
    - src/lyricloop/: core modular package containing engine logic:
        - config.py: global constants and path management.
        - data.py: prompt engineering and dataset preprocessing.
        - environment.py: hardware-aware setup (MPS/CPU/CUDA).
        - metrics.py: inference execution and perplexity scoring.
        - viz.py: standardized plotting and visual utilities.
    - notebooks/: development playground, training workflows, and EDA.
    - reports/: written technical documentation and project summaries.
    - assets/: visual artifacts and plots used in documentation.
    - requirements.txt: dependency management for environment parity.

---

DATA & SOURCE

    - Corpus: 5mm+ Song Lyrics (Genius Dataset).
    - Metadata: Artist mapping via Pitchfork Reviews.
    - Stack: Python, Hugging Face (Transformers, PEFT, TRL), PyTorch, and Google Colab (L4 GPU).

---

EXTERNAL RESOURCES

    - Full Project Workspace (Google Drive): [Access the Notebooks & Raw Data](https://drive.google.com/drive/folders/1M5SJRaaK8OaskUgEsBupgGVN_-fQS3i4?usp=sharing)
    - Training Environment: Google Colab (L4 GPU)

---

STUDIO GUIDE

    - Run on Hugging Face lxtung95/lyricloop
    - App URL: https://lxtung95-lyricloop.hf.space/
        1. Details: Enter a song title and an Artist Aesthetic (e.g., Taylor Swift) to set the tone.
        2. Genre: Select your target genre to adjust rhythmic density.
        3. Compose: Use the Creativity (Temperature) slider to control experimental word choice.
        4. Export: Download the final composition as a .txt file for your creative workflow.

---

SUPPORT

    Visit my GitHub repository for the latest scripts and downloads:
    https://github.com/lxntung95