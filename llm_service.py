import logging
import os
from langchain_ollama import OllamaLLM
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
            ollama_model = os.getenv("OLLAMA_MODEL", "mistral:latest")
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            if not ollama_model or not ollama_base_url:
                logger.error("OLLAMA_MODEL or OLLAMA_BASE_URL not found in environment variables.")
                raise ValueError("OLLAMA_MODEL or OLLAMA_BASE_URL not found in environment variables.")
            
            try:
                logger.info("Initializing OllamaLLM with model: %s, base_url: %s", ollama_model, ollama_base_url)
                cls._instance.llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)
                # Test the LLM with a simple prompt to ensure it's working
                test_response = cls._instance.llm("Hello, are you working?")
                logger.info("LLM initialization successful. Test response: %s", test_response)
            except Exception as e:
                logger.error("Failed to initialize OllamaLLM: %s", e)
                raise RuntimeError(f"Failed to initialize OllamaLLM: {e}")
        return cls._instance

    def get_llm(self):
        # Verify LLM is still responsive before returning
        try:
            test_response = self.llm("Ping")
            logger.debug("LLM ping successful: %s", test_response)
        except Exception as e:
            logger.error("LLM is not responsive: %s", e)
            raise RuntimeError(f"LLM is not responsive: {e}")
        return self.llm

# Global access to the LLM instance
llm_service = LLMService()

def get_llm():
    return llm_service.get_llm()