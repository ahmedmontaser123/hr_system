from abc import ABC , abstractmethod


class LLmInterface(ABC):

    @abstractmethod
    def get_llm(self):
        pass