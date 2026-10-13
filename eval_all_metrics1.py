
import json
import os

from langchain_groq import ChatGroq
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.evaluate import CacheConfig
from deepeval import evaluate


# 1. Configure ChatGroq
groq_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# 2. Create a custom DeepEval model wrapper
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


# 3. Load the original results
with open("result1.jsonl", "r", encoding="utf-8") as file:
    results = [
        json.loads(line)
        for line in file
        if line.strip()
    ]


# 4. Create DeepEval test cases
questions_tests = []

for i, result in enumerate(results, start=1):

    print(f"\nPreparing test {i}/{len(results)}")

    context = result["context"]

    if isinstance(context, str):
        retrieval_context = [context]
    else:
        retrieval_context = context or []

    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result["actual_answer"],
        expected_output=result["expected_answer"],
        retrieval_context=retrieval_context
    )

    questions_tests.append(test_case)


# 5. Configure all four metrics with Groq
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=evaluation_model
)

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=evaluation_model
)

context_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model=evaluation_model
)

correctness = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output is factually "
        "correct based on the expected output. Penalize "
        "contradictions and unsupported factual claims."
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model=evaluation_model
)


# 6. Run batch evaluation
evaluation_results = evaluate(
    test_cases=questions_tests,
    metrics=[
        answer_relevancy,
        faithfulness,
        context_relevancy,
        correctness
    ],
    cache_config=CacheConfig(
        use_cache=False,
        write_cache=False
    )
)


# 7. Extract scores
evaluated_results = []

for original_result, test_result in zip(
    results, evaluation_results.test_results
):

    scores = {
        "answer_relevancy": None,
        "faithfulness": None,
        "context_relevancy": None,
        "correctness": None
    }

    print(f"\nQuestion: {original_result['question']}")

    for metric in test_result.metrics_data or []:

        metric_name = metric.name.strip().lower()

        if metric_name == "answer relevancy":
            scores["answer_relevancy"] = metric.score

        elif metric_name == "faithfulness":
            scores["faithfulness"] = metric.score

        elif metric_name == "contextual relevancy":
            scores["context_relevancy"] = metric.score

        elif metric_name == "correctness":
            scores["correctness"] = metric.score

        print(
            f"{metric.name}: {metric.score} | "
            f"Passed: {metric.success}"
        )

    evaluated_result = {
        **original_result,
        **scores
    }

    evaluated_results.append(evaluated_result)


# 8. Save the evaluated results
with open(
    "evaluated_results1.jsonl",
    "w",
    encoding="utf-8"
) as file:

    for result in evaluated_results:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

print(
    f"\nDone! Evaluated {len(evaluated_results)} test cases."
)
print("Results saved to evaluated_results1.jsonl")
