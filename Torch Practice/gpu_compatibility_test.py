import torch
import logging
from typing import List, Dict
from transformers import AutoModel, AutoConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelFilter:
    def __init__(self, models: List[str]):
        self.models = models
        self.device = self._set_device()
        self.supported_models = []

    def _set_device(self) -> str:
        """Determine available device (GPU or CPU)."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def check_model_compatibility(self) -> List[Dict]:
        """Check if models can be loaded on the current device."""
        results = []
        
        for model_name in self.models:
            try:
                logger.info(f"Testing model: {model_name}")
                
                # Load model configuration
                config = AutoConfig.from_pretrained(model_name)
                
                # Check if model is compatible with device
                is_gpu_compatible = self._is_gpu_compatible(config, model_name)
                
                # Attempt to load model
                load_result = self._test_model_load(model_name)
                
                results.append({
                    "model_name": model_name,
                    "gpu_compatible": is_gpu_compatible,
                    "load_status": load_result["status"],
                    "load_time": load_result["time"],
                    "error": load_result.get("error", None)
                })
                
            except Exception as e:
                logger.error(f"Error processing {model_name}: {str(e)}")
                results.append({
                    "model_name": model_name,
                    "gpu_compatible": False,
                    "load_status": "failed",
                    "load_time": 0.0,
                    "error": str(e)
                })
                
        return results

    def _is_gpu_compatible(self, config, model_name: str) -> bool:
        """Check if model architecture supports GPU acceleration."""
        # Simulated compatibility check based on model size and architecture
        model_size_mb = getattr(config, "model_size_mb", 1000)  # Simulated size
        if self.device == "cuda" and model_size_mb < 4000:  # Arbitrary threshold for demo
            return True
        return False

    def _test_model_load(self, model_name: str) -> Dict:
        """Test loading the model on the current device."""
        import time
        start_time = time.time()
        
        try:
            model = AutoModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            
            # Simulate inference to ensure model is functional
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                model(dummy_input)
                
            return {
                "status": "success",
                "time": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "time": time.time() - start_time,
                "error": str(e)
            }

    def filter_gpu_models(self) -> List[str]:
        """Return list of GPU-compatible models."""
        results = self.check_model_compatibility()
        self.supported_models = [
            result["model_name"] 
            for result in results 
            if result["gpu_compatible"] and result["load_status"] == "success"
        ]
        return self.supported_models

def main():
    # Sample models for testing
    models = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "roberta-large",
        "t5-small"
    ]
    
    # Initialize filter
    model_filter = ModelFilter(models)
    
    # Get device info
    logger.info(f"Running on device: {model_filter.device}")
    
    # Filter GPU-compatible models
    compatible_models = model_filter.filter_gpu_models()
    
    # Log results
    logger.info("GPU-Compatible Models:")
    for model in compatible_models:
        logger.info(f" - {model}")
    
    # Detailed results
    logger.info("\nDetailed Test Results:")
    for result in model_filter.check_model_compatibility():
        logger.info(f"Model: {result['model_name']}")
        logger.info(f"GPU Compatible: {result['gpu_compatible']}")
        logger.info(f"Load Status: {result['load_status']}")
        logger.info(f"Load Time: {result['load_time']:.2f} seconds")
        if result["error"]:
            logger.info(f"Error: {result['error']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()