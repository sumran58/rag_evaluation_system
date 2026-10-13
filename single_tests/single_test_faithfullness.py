
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq
import os


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


evaluation_model = GroqEvalModel(groq_model)


# 3. Create the test case
test_case = LLMTestCase(
    input="Where do penguins live?",
    actual_output=(
        "Penguins live in a variety of environments in the "
        "Southern Hemisphere. They are commonly associated with "
        "the icy coasts of Antarctica, but they also inhabit the "
        "cool shores of South America, southern Africa, and even "
        "the Galapagos Islands near the equator."
    ),
    retrieval_context=[
        "Penguins are flightless seabirds that are specially "
        "adapted for life in the Southern Hemisphere. Although "
        "people often associate them only with Antarctica, "
        "penguins live in a wide range of environments, including "
        "the icy coasts of Antarctica, the cool shores of South "
        "America and southern Africa, and even the Galapagos "
        "Islands near the equator. Their upright posture, compact "
        "bodies, and distinctive black-and-white coloring make "
        "them among the most recognizable birds in the world."
    ]
)


# 4. Configure FaithfulnessMetric
metric = FaithfulnessMetric(
    threshold=0.7,
    model=evaluation_model
)


# 5. Evaluate
metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())