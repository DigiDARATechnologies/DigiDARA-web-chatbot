import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton LLM instance
class LLMService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            # Load configuration from environment variables
            hf_model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
            hf_token = os.getenv("HF_TOKEN", None)

            if not hf_model:
                logger.error("HF_MODEL not found in environment variables.")
                raise ValueError("HF_MODEL not found in environment variables.")

            try:
                logger.info("Initializing Hugging Face model: %s", hf_model)
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model, 
                    token=hf_token,
                    cache_dir="./model_cache"  # Optional: Cache model locally
                )
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model,
                    token=hf_token,
                    device_map="auto",  # Automatically place model on GPU/CPU
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use FP16 for GPU
                    cache_dir="./model_cache"
                )

                # Create a text generation pipeline
                cls._instance.llm = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                    max_new_tokens=100,  # Adjust as needed
                    pad_token_id=tokenizer.eos_token_id
                )

                # Test the LLM with a simple prompt to ensure it's working
                test_response = cls._instance.llm("Hello, are you working?", do_sample=True)[0]["generated_text"]
                logger.info("LLM initialization successful. Test response: %s", test_response)
            except Exception as e:
                logger.error("Failed to initialize Hugging Face model: %s", e)
                raise RuntimeError(f"Failed to initialize Hugging Face model: {e}")
        return cls._instance

    def get_llm(self):
        # Verify LLM is still responsive before returning
        try:
            test_response = self.llm("Ping", do_sample=True)[0]["generated_text"]
            logger.debug("LLM ping successful: %s", test_response)
        except Exception as e:
            logger.error("LLM is not responsive: %s", e)
            raise RuntimeError(f"LLM is not responsive: {e}")
        return self.llm

# Global access to the LLM instance
llm_service = LLMService()

def get_llm():
    return llm_service.get_llm()