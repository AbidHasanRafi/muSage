"""Local LLM module for answer generation using downloaded model weights"""

import logging
import os
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Global model instance (lazy loaded)
_model = None
_tokenizer = None
_model_loaded = False


def _load_model():
    """
    Load the local LLM model (downloads weights on first run).
    Uses a small, CPU-friendly model that can run offline.
    """
    global _model, _tokenizer, _model_loaded
    
    if _model_loaded:
        return _model, _tokenizer
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        # Load model from config (default: Qwen2.5-0.5B-Instruct)
        # This is a lightweight model (~500MB) that runs fast on CPU
        from . import config
        model_name = config.LLM_MODEL_NAME
        
        logger.info(f"Loading local LLM model: {model_name}")
        logger.info("This may take a few minutes on first run (downloading weights)...")
        
        # Load tokenizer and model
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(Path.home() / ".musage" / "models")
        )
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float32,  # Use float32 for CPU (was torch_dtype, now using 'dtype')
            low_cpu_mem_usage=True,  # More efficient loading
            cache_dir=str(Path.home() / ".musage" / "models")
        )
        
        # Move model to CPU explicitly (no device_map to avoid accelerate dependency)
        _model = _model.to('cpu')
        
        # Set to eval mode for inference
        _model.eval()
        
        _model_loaded = True
        logger.info(f"Local LLM loaded successfully: {model_name}")
        
        return _model, _tokenizer
        
    except ImportError as e:
        logger.error(f"Failed to import transformers: {e}")
        logger.error("Install with: pip install transformers torch")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load local LLM: {e}")
        return None, None


def generate_answer(query: str, context: str, max_length: int = 300) -> Optional[str]:
    """
    Generate an answer using the local LLM based on query and context.
    
    Args:
        query: User's question
        context: Relevant information from web search or knowledge base
        max_length: Maximum tokens to generate
    
    Returns:
        Generated answer or None if model unavailable
    """
    model, tokenizer = _load_model()
    
    if model is None or tokenizer is None:
        logger.warning("Local LLM not available, falling back to extraction")
        return None
    
    try:
        import torch
        
        # Construct prompt in chat format
        messages = [
            {
                "role": "system",
                "content": (
                    "You are μSage, a helpful and knowledgeable assistant. "
                    "Answer questions clearly and concisely based on the provided context. "
                    "If the context doesn't contain the answer, say so politely. "
                    "Keep answers focused and informative, typically 2-4 sentences."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a clear, accurate answer based on the context above."
            }
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's answer (after the last user message)
        if "assistant" in full_response.lower():
            parts = full_response.split("assistant")
            answer = parts[-1].strip()
        else:
            # Fallback: take everything after the prompt
            answer = full_response[len(prompt):].strip()
        
        # Clean up
        answer = answer.strip()
        
        # Remove any remaining metadata or tags
        answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        if len(answer) < 20:
            logger.warning("Generated answer too short, returning None")
            return None
        
        logger.info(f"Generated answer using local LLM ({len(answer)} chars)")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer with local LLM: {e}")
        return None


def is_available() -> bool:
    """Check if local LLM is available and loaded."""
    if _model_loaded:
        return _model is not None and _tokenizer is not None
    
    # Try to load if not yet attempted
    try:
        model, tokenizer = _load_model()
        return model is not None and tokenizer is not None
    except:
        return False


def get_model_info() -> dict:
    """Get information about the loaded model."""
    if not _model_loaded:
        return {"status": "not_loaded"}
    
    if _model is None:
        return {"status": "failed"}
    
    from . import config
    return {
        "status": "loaded",
        "model_name": config.LLM_MODEL_NAME,
        "size": "lightweight",
        "device": "CPU"
    }


# Pre-load model on module import (optional, can be disabled for faster startup)
# Uncomment the next line to pre-load model on startup:
# _load_model()
