import google.generativeai as genai
from google.generativeai.types import GenerationConfig

class VLM_GenAI:
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.1, max_output_tokens: int = 1024):
        if not api_key:
            raise ValueError("Gemini API Key is missing.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
