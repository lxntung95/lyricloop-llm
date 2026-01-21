import streamlit as st
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Path Setup: ensure the app can see the modular package
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lyricloop.config import MODEL_ID, RANDOM_STATE
from lyricloop.data import build_inference_prompt, format_lyrics
from lyricloop.metrics import execute_generation
from lyricloop.environment import set_seed

# Page configuration
st.set_page_config(page_title="LyricLoop v2.0", page_icon="🎤", layout="wide")

# Cached model loading
@st.cache_resource
def load_studio_engine():
    """Initializes the Gemma-2b engine for Hugging Face Spaces (CPU)."""
    set_seed(RANDOM_STATE)
    
    # Retrieve the token from Hugging Face Space Secrets
    hf_token = st.secrets["HF_TOKEN"]
    
    # Use the token in the tokenizer and model loading
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # Free Tier uses CPU (forcing float32 for stability)
    device = "cpu"
    dtype = torch.float32
    
    # Load base model skeleton
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device,
        token=hf_token
    )
    
    # Point to the Hugging Face Model Repository for the adapters
    adapter_repo = "lxtung95/lyricloop" 
    
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_repo, 
        token=hf_token
    )
    
    return model, tokenizer

# Studio Interface
st.title("LyricLoop v2.0")
st.caption("Professional AI Songwriting Framework | Powered by Gemma-2b")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("Studio Controls")
creativity = st.sidebar.slider("Creativity (Temperature)", 0.5, 1.2, 0.85)
token_limit = st.sidebar.number_input("Max Tokens", 100, 500, 300)

# Main Input Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Composition Details")
    genre = st.selectbox("Target Genre", ["Pop", "Rock", "Hip-hop", "Electronic", "R&B", "Country"])
    artist = st.text_input("Artist Aesthetic", placeholder="e.g., Taylor Swift")
    title = st.text_input("Song Title", placeholder="Enter your track title...")
    
    generate_btn = st.button("Compose Lyrics", type="primary", use_container_width=True)

with col2:
    st.subheader("Output")
    
    # Create a persistent placeholder for the output
    output_placeholder = st.empty()
    
    if generate_btn:
        with st.spinner("Model is writing..."):
            # Load Engine
            model, tokenizer = load_studio_engine()
            
            # Build Inference Prompt
            prompt = build_inference_prompt(genre, artist, title)
            
            # Generate Lyrics
            raw_output = execute_generation(
                model, tokenizer, prompt, 
                max_tokens=token_limit, 
                temperature=creativity, 
                do_sample=True
            )
            
            # Post-process formatting
            clean_lyrics = format_lyrics(raw_output)
            
            # Update the placeholder specifically
            output_placeholder.text_area(
                "Final Draft", 
                clean_lyrics, 
                height=400, 
                key="lyrics_output"    # adding a key helps mobile state persistence
            )
            
            st.download_button(
                "Export as TXT", 
                clean_lyrics, 
                file_name=f"{title}_lyrics.txt"
            )