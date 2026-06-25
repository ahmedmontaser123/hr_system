from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from schemas import EvaluationSchema

question_prompt = ChatPromptTemplate.from_template("""
You are a senior technical interviewer.

Your task is to generate ONE high-quality interview question.

Rules:
- The question must match the given role and skills ONLY
- Do NOT add explanations
- Do NOT include multiple questions
- Avoid generic questions
- Avoid trivia or vague questions
- The question must be realistic in a real interview setting
- Difficulty must be medium by default

Role:
{role}

Skills:
{description}

Return ONLY valid JSON:

{{
  "question": "..."
}}
""")


classification_prompt = ChatPromptTemplate.from_template("""
You are an interview question classifier.

Classify the question into exactly ONE category.

Allowed categories:
- technical
- problem_solving
- behavioral

Rules:
- Return ONLY one category string
- No explanation
- No extra text

Question:
{question}

Return ONLY valid JSON with one of these exact values:
{{"category": "technical"}}
{{"category": "problem_solving"}}
{{"category": "behavioral"}}
""")


template_message = """
"You are a strict interview evaluator.

Question:
{question}

Answer:
{answer}

Category:
{category}

Rules:
- Be fair.
- Penalize vague answers.
- Reward accuracy.

Output format:
{format_instructions}
"""
evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationSchema)
evaluation_prompt = PromptTemplate(
    template=template_message,
    input_variables=["question", "answer", "category"],
    partial_variables={"format_instructions": evaluation_parser.get_format_instructions()},

    
)




