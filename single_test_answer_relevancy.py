from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase #we provide the test data through this
from deepeval.models import DeepEvalBaseLLM #this is the baseclass that tells deepeval how to communicate witb the model because we are using groq and  if we dot want to do this we can use opnai withot defining all this 
from langchain_groq import ChatGroq


class GroqModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.model.ainvoke(prompt)).content

    def get_model_name(self):
        return "Groq GPT-OSS-120B"


test_case = LLMTestCase(
    input="What do penguins eat?",
    actual_output="Penguins primarily eat krill and tiny fish."
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=GroqModel()
)

metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())