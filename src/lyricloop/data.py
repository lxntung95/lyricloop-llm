import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import RANDOM_STATE, ASSETS_DIR

# -------------------------
# Prompt Construction
# -------------------------

def build_critic_prompt(genre, artist, title, lyrics, max_lyric_length=300):
    """Constructs the instruction-tuning prompt for the Critic persona."""
    lyrics_snippet = lyrics[:max_lyric_length]
    
    instruction = (
        "You are a professional music critic. Provide specific feedback on how to improve "
        "the lyrics based on the genre and artist style. \n"
        "Formatting Rules: \n"
        "1. Use plain text with clear line breaks.\n"
        "2. Ensure all song titles and words have proper spacing."
    )

    context = (
        f"Target Genre: {genre}\n"
        f"Target Artist: {artist}\n"
        f"Target Title: {title}\n\n"
        f"Lyrics to Evaluate:\n{lyrics_snippet}"
    )

    return f"<start_of_turn>user\n{instruction}\n\n{context}<end_of_turn>\n<start_of_turn>model\n"

def build_revision_prompt(genre, artist, title, draft, critiques):
    """Constructs the prompt for the 'Revise' step of the refinement loop."""
    instruction = (
        "You are an expert songwriter. Revise the provided lyrics by incorporating "
        "the specific feedback from the critic while maintaining the genre and artist style."
    )

    context = (
        f"Genre: {genre}\n"
        f"Artist Style: {artist}\n"
        f"Title: {title}\n\n"
        f"Current Draft:\n{draft}\n\n"
        f"Critic Feedback:\n{critiques}"
    )

    return f"<start_of_turn>user\n{instruction}\n\n{context}<end_of_turn>\n<start_of_turn>model\n"

def build_inference_prompt(genre, artist, title):
    """Reconstructs the prompt format used during v1.0 training."""
    instruction = "Generate lyrics for a song based on these details."
    input_context = f"Genre: {genre}\nArtist: {artist}\nTitle: {title}"

    return (
        f"<start_of_turn>user\n{instruction}\n\n{input_context}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def format_prompt(row):
    """Converts a dataframe row into a structured Gemma control-token prompt."""
    instruction = "Generate lyrics for a song based on these details."
    input_context = f"Genre: {row['tag']}\nArtist: {row['artist']}\nTitle: {row['title']}"
    response = row['lyrics']

    return (
        f"<start_of_turn>user\n{instruction}\n\n{input_context}<end_of_turn>\n"
        f"<start_of_turn>model\n{response}<end_of_turn>"
    )

# -------------------------
# Text Processing
# -------------------------

def format_lyrics(text):
    """Cleans up raw model output by enforcing structural newlines and spacing."""
    # Add double newlines before section headers like [Verse], [Chorus]
    text = re.sub(r'(\[.*?\])', r'\n\n\1\n', text)

    # Add a newline when a capital letter follows a lowercase letter immediately
    text = re.sub(r'([a-z])([A-Z])', r'\1\n\2', text)
    return text.strip()

# -------------------------
# Dataset Management
# -------------------------

def format_critic_training_row(row):
    """Standardizes raw rows into the Critic instruction-tuning format."""
    prompt = build_critic_prompt(row.tag, row.artist, row.title, row.lyrics)
    
    target_output = (
        f"Genre Fit: The {row.tag} style is well-maintained.\n"
        f"Artist Style: Matches the {row.artist} aesthetic.\n"
        f"Improvements: Consider refining the rhythmic flow in the second verse."
    )

    return f"{prompt}{target_output}<eos>"

def prepare_lyric_dataset(lyrics_filename, reviews_filename, songs_per_genre=200):
    """Loads, cleans, and balances the dataset while exporting EDA plots."""
    from .viz import save_figure
    
    lyrics_path = os.path.join("data", lyrics_filename)
    reviews_path = os.path.join("data", reviews_filename)

    print(f"Loading & Cleaning Raw Data...")
    
    lyrics_df = pd.read_csv(lyrics_path, on_bad_lines='skip')
    reviews_df = pd.read_csv(reviews_path)

    lyrics_df = lyrics_df.dropna(subset=['lyrics', 'artist', 'tag'])
    reviews_df = reviews_df.dropna(subset=['genre', 'artist'])

    lyrics_clean = lyrics_df.drop_duplicates(subset="artist")[["artist", "lyrics", "title", "tag"]]
    merged_df = reviews_df.merge(lyrics_clean, on="artist", how="left").dropna(subset=["lyrics", "tag"])

    # --- Plot 1: Raw Distribution ("Before") ---
    plt.figure(figsize=(10, 5))
    top_raw = merged_df['tag'].value_counts().nlargest(10)
    sns.barplot(x=top_raw.values, y=top_raw.index, hue=top_raw.index, palette='viridis', legend=False)
    plt.title(f"Raw Genre Distribution (n={len(merged_df):,})")
    save_figure("eda_1_raw_distribution.png")

    # Class balancing logic
    balanced_df = merged_df.groupby("tag", group_keys=False).apply(
        lambda x: x.sample(min(len(x), songs_per_genre), random_state=RANDOM_STATE)
    )

    # --- Plot 2: Balanced Distribution ("After") ---
    plt.figure(figsize=(10, 5))
    sns.countplot(data=balanced_df, y='tag', hue='tag', palette='magma', legend=False)
    plt.title(f"Balanced Genre Distribution (n={len(balanced_df):,})")
    save_figure("eda_2_balanced_distribution.png")

    return balanced_df