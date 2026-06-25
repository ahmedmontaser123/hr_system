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
You are an expert technical interviewer.

Evaluate the candidate answer carefully.

Question:
{question}

Answer:
{answer}

Category:
{category}

Scoring Rubric (0-10):

Technical Questions:
- 0-2: Wrong answer or completely irrelevant.
- 3-4: Very limited understanding, major mistakes.
- 5-6: Partially correct, misses important concepts.
- 7-8: Mostly correct, minor missing details.
- 9-10: Accurate, complete, and demonstrates strong understanding.

Behavioral Questions:
- 0-2: No meaningful answer.
- 3-4: Vague answer without examples.
- 5-6: Acceptable answer but lacks depth.
- 7-8: Good example with clear reasoning.
- 9-10: Strong structured answer with measurable impact.

Evaluation Steps:
1. Check if the answer addresses the question.
2. Check technical correctness.
3. Check completeness.
4. Check clarity and communication.
5. Give a final score from 0 to 10.

Important Rules:
- Never give high scores for vague answers.
- Reward accurate technical details.
- Reward real examples when applicable.
- If the answer is unrelated, score <= 2.
- If the answer is partially correct, score between 4 and 6.
- Explain briefly why the score was assigned.

Output format:
{format_instructions}
"""

evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationSchema)
evaluation_prompt = PromptTemplate(
    template=template_message,
    input_variables=["question", "answer", "category"],
    partial_variables={"format_instructions": evaluation_parser.get_format_instructions()},

    
)




