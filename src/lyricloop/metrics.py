import torch
import numpy as np

# -------------------------
# Generation Engines
# -------------------------

def execute_generation(model, tokenizer, prompt, max_tokens=300, temperature=0.85, do_sample=False):
    """
    A universal engine that handles GPU movement, sampling, and decoding.
    The do_sample=False default is ideal for objective Critic tasks.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            no_repeat_ngram_size=3,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return generated_text.strip()

def get_token_confidences(model, tokenizer, prompt, max_tokens=50):
    """
    Generates text and returns a list of (token, confidence_score) tuples.
    Used for creating confidence heatmaps in the UI.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id
        )

    input_len = inputs.input_ids.shape[1]
    gen_ids = outputs.sequences[0][input_len:]

    # Calculate softmax probabilities for each generated token
    probs = [torch.softmax(score, dim=-1)[0, tid].item() for tid, score in zip(gen_ids, outputs.scores)]
    tokens = [tokenizer.decode(tid) for tid in gen_ids]

    return list(zip(tokens, probs))

# -------------------------
# Evaluation Metrics
# -------------------------

def calculate_perplexity(model, tokenizer, text):
    """
    Computes the perplexity (uncertainty) of the model for a specific text sequence.
    Lower score = the model finds the text natural/predictable.
    Higher score = the model finds the text confusing/alien.
    """
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(inputs.input_ids, labels=inputs.input_ids)
        loss = outputs.loss

    # Perplexity is mathematically the exponential of the cross-entropy loss
    return torch.exp(loss).item()

# -------------------------
# Trainer Log Parsers
# -------------------------

def extract_trainer_metrics(model_trainer):
    """
    Universal log parser for Hugging Face Trainer.
    Extracts step-by-step history for plotting and final validation.
    """
    logs = model_trainer.state.log_history

    # Extract coordinates for plotting (Training vs Evaluation)
    train_metrics = [{"step": x["step"], "loss": x["loss"]} for x in logs if "loss" in x]
    eval_metrics = [{"step": x["step"], "loss": x["eval_loss"]} for x in logs if "eval_loss" in x]

    final_loss = eval_metrics[-1]["loss"] if eval_metrics else None

    return {
        "train_steps": [x["step"] for x in train_metrics],
        "train_loss": [x["loss"] for x in train_metrics],
        "eval_steps": [x["step"] for x in eval_metrics],
        "eval_loss": [x["loss"] for x in eval_metrics],
        "val_loss": final_loss,
        "perplexity": np.exp(final_loss) if final_loss else None
    }