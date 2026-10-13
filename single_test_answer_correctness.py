from deepeval.test_case import LLMTestCase,SingleTurnParams
from deepeval.metrics import GEval
import json 
from langchain_groq import ChatGroq
import os
from deepeval.models import DeepEvalBaseLLM

# 1. Configure ChatGroq
groq_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Create a custom DeepEval model
class GroqEvalModel(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Groq GPT OSS 120B"


# 3. Initialize the evaluator
evaluation_model = GroqEvalModel(groq_model)

test_case=LLMTestCase(
    input="Where do penguins live?",
    actual_output="Penguins live in a variety of environments in the Southern Hemisphere. They are commonly associated with the icy coasts of Antarctica, but they also inhabit the cool shores of South America, southern Africa, and even the Galapagos Islands near the equator.",
    expected_output="Penguins live in the Southern Hemisphere, including Antarctica, South America, southern Africa, and the Galapagos Islands."
)

metric=GEval(
    name='correctness',
    criteria=("determine whther the actual output is factually correct based on the expected output"),
    evaluation_params=[
        #this tells depeval which piece of info the llm judge need to eval 
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
        
    ],
    model=evaluation_model

)
metric.measure(test_case=test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)