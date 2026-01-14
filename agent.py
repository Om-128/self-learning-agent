import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from langchain.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Define tools
''' Create Wikipedia tool '''
wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki)

''' Create tavily search tool '''
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearch(max_results=5)

# Prompt
''' Define Prompt '''

SYSTEM_PROMPT = """
You are an analytical agent.

Rules:
1. Answer the question.
2. Use the available tools.
3. Explicitly say which tools were used.
4. If unsure, say why you are unsure.
5. Never fabricate facts.

Give answer in this Format:
---------------------------
ANSWER:
Write your answer

TOOL USAGE:
Write tools name you used

Evaluation:
What mistakes you made
what do you learned from your mistakes
"""



''' Define Agent '''
llm = ChatOllama(model="llama3.1:8b",temperature=0.7)

agent = create_agent(
    model=llm,
    tools=[search_tool], 
    system_prompt=SYSTEM_PROMPT
)


question = input("Enter your query: ")
answer = agent.invoke({"messages": question})

print("\n Orignal Answer" + answer["messages"][-1].content)

print("=====================================================")

EVALUATION_PROMPT = """
You are now acting as an evaluator agent.

You are given:
1. A USER QUESTION
2. The AGENT'S ANSWER

Your task:
1. Verify whether the answer is factually correct.
2. Decide whether the correct tool was used (Wikipedia / Tavily / none).
3. If incorrect, explain WHY.
4. If the answer is partially correct, explain what is missing.
5. Suggest what should have been done instead.

You MUST verify facts using tools if needed.

--------------------
QUESTION:
{question}

AGENT ANSWER:
{answer}

--------------------
Respond in this exact format:

EVALUATION:
- Is the answer correct? (Yes / No)
- Was the correct tool used?
- Explanation of mistakes (if any):

CORRECTED ANSWER:
If needed, write the corrected answer here. If not, say "No correction needed."
"""

evaluation = agent.invoke({
    "input":EVALUATION_PROMPT.format(
        question=question,
        answer=answer["messages"][-1].content
    )
})


print("\n=== EVALUATION ===")
print(evaluation["messages"][-1].content)