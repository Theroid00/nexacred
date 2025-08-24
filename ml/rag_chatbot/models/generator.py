"""
IBM Granite Model Generator
===========================

Loads and manages IBM Granite 3.1 8B Instruct model for text generation.
Automatically downloads model on first use and caches for subsequent runs.
"""

import torch
import os
from pathlib import Path
from typing import Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download, HfFolder
import logging

logger = logging.getLogger(__name__)


def check_model_exists(model_id: str) -> bool:
    """
    Check if model is already downloaded and cached locally.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        True if model exists locally, False otherwise
    """
    try:
        # Check if model files exist in HuggingFace cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_name = f"models--{model_id.replace('/', '--')}"
        model_cache_path = os.path.join(cache_dir, model_cache_name)
        
        if os.path.exists(model_cache_path):
            # Check for essential model files
            essential_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            
            # Look for files in snapshots subdirectory
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                for snapshot in os.listdir(snapshots_dir):
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if os.path.isdir(snapshot_path):
                        files_in_snapshot = os.listdir(snapshot_path)
                        # Check if we have at least some essential files
                        if any(f in files_in_snapshot for f in essential_files):
                            logger.info(f"Model {model_id} found in local cache")
                            return True
        
        logger.info(f"Model {model_id} not found in local cache, will download")
        return False
        
    except Exception as e:
        logger.warning(f"Error checking model cache: {e}")
        return False


def estimate_model_size(model_id: str) -> str:
    """
    Estimate model download size for user information.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        Estimated size string
    """
    if "8b" in model_id.lower():
        return "~16GB"
    elif "3b" in model_id.lower():
        return "~6GB"
    elif "1b" in model_id.lower():
        return "~2GB"
    else:
        return "~10-20GB"


def download_model_with_progress(model_id: str) -> str:
    """
    Download model with progress information.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        Local path to downloaded model
    """
    try:
        estimated_size = estimate_model_size(model_id)
        logger.info(f"📥 Downloading {model_id} (estimated size: {estimated_size})")
        logger.info("⏳ This may take several minutes depending on your internet connection...")
        
        # Download model to cache
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=None,  # Use default cache directory
            resume_download=True,  # Resume if interrupted
            local_files_only=False
        )
        
        logger.info(f"✅ Model downloaded successfully to: {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        raise RuntimeError(f"Model download failed: {e}") from e


def load_generator(
    model_id: str, 
    device_map: str = "auto", 
    use_4bit: bool = False,
    force_download: bool = False
) -> Tuple[Any, Any]:
    """
    Load IBM Granite tokenizer and model with intelligent caching.
    
    On first run: Downloads and caches the model (~16GB for Granite 3.1 8B)
    On subsequent runs: Loads from local cache (much faster)
    
    Args:
        model_id: Hugging Face model identifier
        device_map: Device mapping strategy
        use_4bit: Enable 4-bit quantization
        force_download: Force re-download even if cached
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        RuntimeError: If model loading fails
    """
    
    try:
        # Check if model exists locally (unless forcing download)
        if not force_download and check_model_exists(model_id):
            logger.info(f"🔄 Loading cached model: {model_id}")
        else:
            if force_download:
                logger.info(f"🔄 Force downloading model: {model_id}")
            
            # Download model if not cached
            download_model_with_progress(model_id)
        
        logger.info(f"📚 Loading tokenizer for model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=False,  # Allow download if needed
            trust_remote_code=True   # Required for some models
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")
        
        # Configure model loading parameters
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "local_files_only": False,
            "trust_remote_code": True
        }
        
        # Handle 4-bit quantization
        if use_4bit:
            try:
                from bitsandbytes import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                logger.info("✅ Enabled 4-bit quantization")
                
            except ImportError:
                logger.warning(
                    "⚠️ bitsandbytes not available, falling back to non-quantized loading"
                )
            except Exception as e:
                logger.warning(f"⚠️ 4-bit quantization failed: {e}, falling back")
        
        logger.info(f"🤖 Loading model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        # Log model info
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📍 Device: {device}")
        logger.info(f"🔧 Data type: {dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        error_msg = f"Failed to load model '{model_id}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_generator_from_config(config, force_download: bool = False) -> Tuple[Any, Any]:
    """
    Load generator using configuration object.
    
    Args:
        config: Configuration object with model settings
        force_download: Force re-download even if cached
        
    Returns:
        Tuple of (model, tokenizer)
    """
    return load_generator(
        model_id=config.granite_model_id,
        device_map=config.device_map,
        use_4bit=config.use_4bit,
        force_download=force_download
    )


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """
    Generate text using the loaded model.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Generated text (assistant portion only)
    """
    
    try:
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Conservative max length
        )
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decode full output
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated portion (after the prompt)
        assistant_text = full_text[len(prompt):].strip()
        
        logger.debug(f"Generated {len(assistant_text)} characters")
        return assistant_text
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error during generation: {str(e)}"


if __name__ == "__main__":
    """
    Demo/test script for the generator.
    Run this to test model downloading and generation.
    """
    import sys
    from pathlib import Path
    
    # Add parent directory for config import
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from config import Config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        print("🚀 IBM Granite 3.1 8B Instruct Generator Test")
        print("=" * 50)
        
        # Load configuration
        config = Config()
        print(f"📋 Model ID: {config.granite_model_id}")
        
        # Check if model exists
        if check_model_exists(config.granite_model_id):
            print("✅ Model found in cache")
        else:
            print("📥 Model will be downloaded on first load")
            estimated_size = estimate_model_size(config.granite_model_id)
            print(f"📊 Estimated download size: {estimated_size}")
            
            response = input("Continue with download? (y/n): ").lower().strip()
            if response != 'y':
                print("❌ Download cancelled")
                sys.exit(0)
        
        # Load model
        print("\n🔄 Loading model...")
        model, tokenizer = load_generator_from_config(config)
        
        # Test generation
        print("\n💬 Testing generation...")
        test_prompt = "What are the key features of peer-to-peer lending in India?"
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_new_tokens=150,
            temperature=0.7
        )
        
        print(f"\n📝 Prompt: {test_prompt}")
        print(f"🤖 Response: {response}")
        print("\n✅ Generator test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logging.error("Test failed", exc_info=True)
        sys.exit(1)
