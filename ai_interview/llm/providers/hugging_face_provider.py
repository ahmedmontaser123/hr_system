from llm.llm_interface import LLMInterface
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from helpers import Settings

class HuggingFaceProvider(LLMInterface):
    def __init__(self, settings: Settings):
        self.model_id = settings.LLM_MODEL_NAME
        self.llm = self._load()

    def _load(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="cpu"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
        )
        return HuggingFacePipeline(pipeline=pipe)

    def get_llm(self):
        return self.llm