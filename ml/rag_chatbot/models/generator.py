"""
IBM Granite Model Generator
===========================

Loads and manages IBM Granite 3.3 2B Instruct model for text generation.
Automatically downloads model on first use and caches for subsequent runs.
"""

import torch
import os
import psutil
import gc
from pathlib import Path
from typing import Tuple, Any, Optional, List, Dict
import logging

# Fix imports - handle missing dependencies gracefully
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from huggingface_hub import snapshot_download, HfFolder
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Missing dependencies. Please install: pip install torch transformers sentence-transformers scikit-learn")

logger = logging.getLogger(__name__)


def get_system_memory_gb() -> float:
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024**3)


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            return 0.0
    return 0.0


def cleanup_model_memory(model: Optional[Any] = None, tokenizer: Optional[Any] = None):
    """Clean up model memory and force garbage collection."""
    try:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Model memory cleanup completed")

    except Exception as e:
        logger.warning(f"Error during memory cleanup: {e}")


def validate_model_requirements(model_id: str, config) -> bool:
    """
    Validate if system can handle the model based on memory requirements.

    Args:
        model_id: Model identifier
        config: Configuration object

    Returns:
        True if system can handle the model
    """
    try:
        # Updated estimates for IBM Granite 3.3 2B and other models
        if "granite-3.3-2b" in model_id.lower():
            estimated_memory_gb = 4.0 if config.use_4bit else 8.0  # 2B model is much smaller
        elif "8b" in model_id.lower() or "8B" in model_id:
            estimated_memory_gb = 16.0
        elif "3b" in model_id.lower() or "3B" in model_id:
            estimated_memory_gb = 6.0
        elif "2b" in model_id.lower() or "2B" in model_id:
            estimated_memory_gb = 4.0
        elif "1b" in model_id.lower() or "1B" in model_id:
            estimated_memory_gb = 2.0
        else:
            estimated_memory_gb = 4.0  # Conservative estimate

        # Check system memory
        available_memory = get_system_memory_gb()

        if config.enable_gpu and torch.cuda.is_available():
            gpu_memory = get_gpu_memory_gb()
            if gpu_memory < estimated_memory_gb:
                logger.warning(f"GPU memory ({gpu_memory:.1f}GB) insufficient for model ({estimated_memory_gb:.1f}GB)")
                return False
        else:
            if available_memory < estimated_memory_gb:
                logger.warning(f"System memory ({available_memory:.1f}GB) insufficient for model ({estimated_memory_gb:.1f}GB)")
                return False

        logger.info(f"Memory validation passed for {model_id}: {estimated_memory_gb:.1f}GB required")
        return True

    except Exception as e:
        logger.warning(f"Could not validate memory requirements: {e}")
        return True  # Assume it's fine if we can't check


def try_load_model_with_fallbacks(config, force_download: bool = False) -> Tuple[Any, Any]:
    """
    Try to load the primary model, falling back to alternatives if it fails.

    Args:
        config: Configuration object with model settings
        force_download: Force re-download even if cached

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        RuntimeError: If all models fail to load
    """
    models_to_try = [config.granite_model_id] + config.granite_model_fallbacks

    for model_id in models_to_try:
        try:
            logger.info(f"Attempting to load model: {model_id}")

            # Validate memory requirements
            if not validate_model_requirements(model_id, config):
                logger.warning(f"Skipping {model_id} due to memory constraints")
                continue

            # Try to load the model
            model, tokenizer = load_generator(
                model_id=model_id,
                device_map=config.device_map,
                use_4bit=config.use_4bit,
                force_download=force_download,
                max_memory_gb=config.max_memory_gb,
                enable_gpu=config.enable_gpu
            )

            logger.info(f"Successfully loaded model: {model_id}")
            return model, tokenizer

        except Exception as e:
            logger.warning(f"Failed to load {model_id}: {e}")
            # Clean up any partial loading
            cleanup_model_memory()
            continue

    raise RuntimeError(f"All models failed to load. Tried: {models_to_try}")


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
    if "granite-3.3-2b" in model_id.lower():
        return "~4GB"  # IBM Granite 3.3 2B specific
    elif "8b" in model_id.lower():
        return "~16GB"
    elif "3b" in model_id.lower():
        return "~6GB"
    elif "2b" in model_id.lower():
        return "~4GB"
    elif "1b" in model_id.lower():
        return "~2GB"
    else:
        return "~4-8GB"  # Conservative estimate for unknown models


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
    model_id: str = "ibm-granite/granite-3.3-2b-instruct",
    device_map: str = "auto",
    use_4bit: bool = True,
    force_download: bool = False,
    max_memory_gb: float = 6.0,
    enable_gpu: bool = True
) -> Tuple[Any, Any]:
    """
    Load IBM Granite 3.3 2B Instruct model with optimized settings.

    Args:
        model_id: Hugging Face model identifier
        device_map: Device mapping strategy
        use_4bit: Enable 4-bit quantization for efficiency
        force_download: Force re-download even if cached
        max_memory_gb: Maximum memory usage
        enable_gpu: Enable GPU usage if available

    Returns:
        Tuple of (model, tokenizer)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Missing dependencies. Install: pip install torch transformers sentence-transformers")

    try:
        logger.info(f"Loading IBM Granite model: {model_id}")

        # Check if model exists locally
        if not force_download and check_model_exists(model_id):
            logger.info(f"Model {model_id} found in cache")
        else:
            logger.info(f"Downloading model {model_id} - this may take a while...")
            size_estimate = estimate_model_size(model_id)
            logger.info(f"Estimated download size: {size_estimate}")

        # Configure quantization for efficiency
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )

        # Load tokenizer first
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True
        )

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure device and memory
        device = "cuda" if enable_gpu and torch.cuda.is_available() else "cpu"

        # Load model with optimized settings for Granite 3.3 2B
        logger.info(f"Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map if enable_gpu else None,
            torch_dtype=torch.float16 if enable_gpu else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True
        )

        # Move to device if not using device_map
        if device_map is None:
            model = model.to(device)

        # Enable evaluation mode
        model.eval()

        logger.info(f"Successfully loaded {model_id}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        cleanup_model_memory()
        raise


def load_generator_from_config(config, force_download: bool = False) -> Tuple[Any, Any]:
    """
    Load generator using configuration object with fallback support.

    Args:
        config: Configuration object with model settings
        force_download: Force re-download even if cached
        
    Returns:
        Tuple of (model, tokenizer)
    """
    return try_load_model_with_fallbacks(config, force_download)


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    do_sample: bool = True,
    pad_token_id: Optional[int] = None
) -> str:
    """
    Generate text using IBM Granite 3.3 2B model with optimized parameters.

    Args:
        model: Loaded Granite model
        tokenizer: Loaded tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        pad_token_id: Padding token ID

    Returns:
        Generated text response
    """
    try:
        # Prepare inputs
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Granite 3.3 2B context length
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Set pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        # Generate with optimized parameters for Granite
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True
            )

        # Decode response
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "I apologize, but I encountered an error generating a response. Please try again."


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
