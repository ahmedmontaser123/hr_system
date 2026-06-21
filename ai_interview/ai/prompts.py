from langchain_core.prompts import ChatPromptTemplate
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


question_generation_prompt = ChatPromptTemplate.from_template("""
You are a senior technical interviewer conducting a professional job interview.

Role:
{role}

Job Description:
{description}

Target Topic:
{topic}

Difficulty:
{difficulty}

Previously Asked Topics:
{previous_topics}

Instructions:
- Generate exactly ONE interview question.
- The question must be directly related to the target topic.
- The question must align with the role and job description.
- Avoid repeating previously covered topics.
- Prefer practical and scenario-based questions.
- Avoid yes/no questions.
- Avoid generic textbook questions.
- Difficulty must match the requested level.
- Return ONLY the question text.
- Do not include explanations, numbering, markdown, or answers.

Interview Question:
""")


next_question_decision_prompt = ChatPromptTemplate.from_template("""
You are a senior technical interviewer.

Analyze the candidate's performance and decide the next interview step.

Role:
{role}

Current Question:
{question}

Candidate Answer:
{answer}

Evaluation Result:
{evaluation}

Topics Already Covered:
{previous_topics}

Instructions:

- If the candidate performed poorly, stay in the same topic and reduce difficulty if needed.
- If the candidate performed moderately, stay in the same topic and ask a deeper follow-up question.
- If the candidate performed well, move to a new relevant topic required by the job description.
- Do not select a topic that has already been sufficiently covered.
- Focus on maximizing assessment coverage of the candidate's skills.

Return ONLY valid JSON in the following format:

{{
  "action": "same_topic | follow_up | new_topic",
  "next_topic": "topic name",
  "difficulty": "Easy | Medium | Hard",
  "reason": "short explanation"
}}
""")