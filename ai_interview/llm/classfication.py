from .chains import Chains
class ClassficationQuestion:
    def __init__(self, chains:Chains):
        self.classfication_chain = chains.classfication_chain

    
    def classify(self, question: str):
        try:
          result = self.classfication_chain.invoke({
            "question": question
            })
          return result.category
        except Exception:
              return "technical"  # fallback لو فشل