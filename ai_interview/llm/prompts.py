from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

# prompts
relevance_prompt = ChatPromptTemplate.from_template("""
Check if answer is related to question.

Be flexible.
Accept partial relevance.

Question:
{question}

Answer:
{answer}

Return JSON:
{{
  "score": 0-10,
  "reason": ""
}}
""")



classification_prompt = ChatPromptTemplate.from_template("""
You are an AI HR assistant.

Classify the following interview question.

Question:
{question}

Possible categories:
- technical
- problem_solving
- behavioral

Rules:
- Return only ONE category
- Return JSON only

Output format:
{{
  "type": ""
}}
""")


technical_prompt =ChatPromptTemplate.from_template("""

You are a technical interviewer.

Question:
{question}

Answer:
{answer}

Evaluate:
- correctness
- depth
- technical terminology

Return JSON:
{{
  "correctness": int,
  "depth": int,
  "terminology": int,
  "feedback": str
}}
"""
)


problem_prompt = ChatPromptTemplate.from_template("""
You are a technical interviewer.

Question:
{question}

Answer:
{answer}

Evaluate:
- logic
- step thinking
- decomposition
- effectiveness
- tradeoffs

Return JSON:
{{
  "logic": int,
  "step_thinking": int,
  "decomposition": int,
  "effectiveness": int,
  "tradeoffs": int,
  "feedback": str
}}
"""
)

behavioral_prompt =ChatPromptTemplate.from_template("""
You are an HR interviewer.

Question:
{question}

Answer:
{answer}

Evaluate:
- communication
- teamwork
- leadership
- emotional intelligence

Return JSON:
{{
  "communication": int,
  "teamwork": int,
  "leadership": int,
  "emotional_intelligence": int,
  "feedback": str
}}
"""
)




question_prompt = PromptTemplate(
    input_variables=["role", "description"],
    template="""
You are a technical interviewer.

Generate one technical interview question based on:

Role: {role}

Job Description:
{description}

Return only the question.
"""
)


