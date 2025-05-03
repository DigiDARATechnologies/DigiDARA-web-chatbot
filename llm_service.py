from langchain_ollama import OllamaLLM

# Singleton LLM instance
class LLMService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance.llm = OllamaLLM(model="mistral:latest", base_url="http://localhost:11434")
        return cls._instance

    def get_llm(self):
        return self.llm

# Global access to the LLM instance
llm_service = LLMService()

def get_llm():
    return llm_service.get_llm()