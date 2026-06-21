# chains.py
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from . import prompts

def build_chains(llm) -> dict:
    json_parser = JsonOutputParser()
    str_parser = StrOutputParser()

    return {
        "relevance": prompts.relevance_prompt | llm | json_parser,
        "classification": prompts.classification_prompt | llm | json_parser,
        "technical": prompts.technical_prompt | llm | json_parser,
        "problem": prompts.problem_prompt | llm | json_parser,
        "behavioral": prompts.behavioral_prompt | llm | json_parser,
        "question_generation": prompts.question_prompt | llm | str_parser,
    }