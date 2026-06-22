from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    FILE_EXTENTION: list
    FILE_SIZE: int
    MODEL_TYPE: str
    DEVICE: str
    COMPUTE_TYPE: str
    LLM_MODEL_NAME: str 
    LLM_MAX_NEW_TOKENS:int = 250
    LLM_TEMPERATURE: float = 0.6
    OLLAMA_MODEL:str

    class Config:
        env_file = ".env" 

def get_settings():
    return Settings()
