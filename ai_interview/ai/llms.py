from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from helpers import Settings
from ai import BaseLoader


_model_cache = None
_tokenizer_cache = None

class LLMLoader(BaseLoader):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model_id = settings.LLM_MODEL_NAME
        self.llm = self._load()

    def _load(self):
        global _model_cache, _tokenizer_cache

        if _model_cache is None:
            _tokenizer_cache = AutoTokenizer.from_pretrained(self.model_id)
            _model_cache = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map=self.settings.DEVICE,
            )

        pipe = pipeline(
            "text-generation",
            model=_model_cache,
            tokenizer=_tokenizer_cache,
            max_new_tokens=self.settings.LLM_MAX_NEW_TOKENS,
        )
        return HuggingFacePipeline(pipeline=pipe)
    
    
    def get_llm(self):
        return self.llm
