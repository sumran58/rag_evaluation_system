
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq

import RAG1.tools as t1
from RAG1.tools import collecting_info
from RAG1.prompts import system_agent_prompt

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

agent = create_agent(
    model=llm,
    system_prompt=system_agent_prompt,
    tools=[collecting_info]
)

def ask(question: str) -> dict:
    """Answer the user's question about the PDF."""

    t1.last_retrieved_context = []

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    answer = result["messages"][-1].content

    return {
        "answer": answer,
        "context": t1.last_retrieved_context
    }

if __name__ == "__main__":
    print(ask("What is the diet of penguins"))
