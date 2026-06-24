from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

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

evaluator_prompt = ChatPromptTemplate.from_template("""
You are a strict senior interview evaluator.

You MUST evaluate fairly and consistently.

Inputs:
- Question
- Answer
- Category

Evaluation Rules:
- Do NOT be overly generous
- Do NOT assume missing information
- Penalize vague or generic answers
- Reward clarity, correctness, and depth
- If answer is wrong, score must be low (≤4)
- If answer is partially correct, score 5-7
- If excellent, score 8-10

Category-specific logic:
- technical → correctness, depth, accuracy
- problem_solving → logic, structure, efficiency
- behavioral → clarity, communication, realism

Question:
{question}

Answer:
{answer}

Category:
{category}

Return ONLY valid JSON:

{{
  "score": 0-10,
  "feedback": "Short but precise explanation of strengths and weaknesses"
}}
""")

