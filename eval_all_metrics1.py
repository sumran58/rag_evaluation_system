
import os
import json
import time
import asyncio
import random
from pathlib import Path

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_groq import ChatGroq

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import CacheConfig


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

INPUT_FILE = Path("result1.jsonl")
OUTPUT_FILE = Path("evaluated_results1.jsonl")

BATCH_SIZE = 1
MAX_ATTEMPTS = 6

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file."
    )


# --------------------------------------------------
# 2. Initialize Groq
# --------------------------------------------------

groq_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=0,
)

# Shared lock: serialize calls made through this wrapper.
groq_request_lock = asyncio.Lock()


# --------------------------------------------------
# 3. DeepEval model wrapper with retry handling
# --------------------------------------------------

class GroqEvalModel(DeepEvalBaseLLM):

    def __init__(self, model, max_attempts=6):
        self.model = model
        self.max_attempts = max_attempts

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        for attempt in range(self.max_attempts):
            try:
                response = self.model.invoke(prompt)
                return response.content

            except RateLimitError:
                if attempt == self.max_attempts - 1:
                    raise

                delay = min(5 * (2 ** attempt), 60)
                delay += random.uniform(0, 2)

                print(
                    f"Groq rate limit reached. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                time.sleep(delay)

        raise RuntimeError("Groq retry attempts exhausted")

    async def a_generate(self, prompt: str) -> str:
        for attempt in range(self.max_attempts):
            try:
                # Prevent overlapping requests through this wrapper.
                async with groq_request_lock:
                    response = await self.model.ainvoke(prompt)

                return response.content

            except RateLimitError:
                if attempt == self.max_attempts - 1:
                    raise

                delay = min(5 * (2 ** attempt), 60)
                delay += random.uniform(0, 2)

                print(
                    f"Groq rate limit reached. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Groq retry attempts exhausted")

    def get_model_name(self):
        return "Groq GPT OSS 20B"


groq_eval_model = GroqEvalModel(groq_model)


# --------------------------------------------------
# 4. Define metrics
# --------------------------------------------------

answer_relevancy = AnswerRelevancyMetric(
    model=groq_eval_model,
    async_mode=False,
)

faithfulness = FaithfulnessMetric(
    model=groq_eval_model,
    async_mode=False,
)

context_relevancy = ContextualRelevancyMetric(
    model=groq_eval_model,
    async_mode=False,
)

correctness = GEval(
    name="Correctness",
    criteria=(
        "Evaluate whether the actual answer accurately answers "
        "the question compared with the expected answer. Penalize "
        "factual errors, contradictions, and missing important "
        "information."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=groq_eval_model,
    async_mode=False,
)

metrics = [
    answer_relevancy,
    faithfulness
    
]


# --------------------------------------------------
# 5. Load input JSONL
# --------------------------------------------------

records = []

with INPUT_FILE.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number}: {exc}"
            ) from exc

        required_fields = [
            "question",
            "actual_answer",
            "expected_answer",
            "context",
        ]

        missing = [
            field for field in required_fields
            if field not in record
        ]

        if missing:
            raise ValueError(
                f"Line {line_number} is missing fields: {missing}"
            )

        records.append(record)

if not records:
    raise ValueError(f"No records found in {INPUT_FILE}")

print(f"Loaded {len(records)} test cases.")


# --------------------------------------------------
# 6. Build DeepEval test cases
# --------------------------------------------------

test_cases = []

for record in records:
    context = record["context"]

    if isinstance(context, str):
        retrieval_context = [context]
    elif isinstance(context, list):
        retrieval_context = context
    else:
        raise ValueError(
            "Each record's context must be a string or list."
        )

    test_cases.append(
        LLMTestCase(
            input=record["question"],
            actual_output=record["actual_answer"],
            expected_output=record["expected_answer"],
            retrieval_context=retrieval_context,
        )
    )


# --------------------------------------------------
# 7. Evaluate in batches and save completed results
# --------------------------------------------------

# Start a fresh output file for this run.
OUTPUT_FILE.write_text("", encoding="utf-8")

completed = 0
total = len(test_cases)

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)

    batch_records = records[start:end]
    batch_cases = test_cases[start:end]

    print(
        f"\nEvaluating test cases {start + 1}-{end} "
        f"of {total}..."
    )

    try:
        evaluation_results = evaluate(
            test_cases=batch_cases,
            metrics=metrics,
            cache_config=CacheConfig(
                use_cache=False,
                write_cache=False,
            ),
        )

        batch_output = []

        for index, test_result in enumerate(
            evaluation_results.test_results
        ):
            result_record = dict(batch_records[index])
            scores = {}

            for metric_result in test_result.metrics_data:
                scores[metric_result.name] = {
                    "score": metric_result.score,
                    "success": metric_result.success,
                    "reason": metric_result.reason,
                }

            result_record["evaluation_scores"] = scores
            batch_output.append(result_record)

        # Append only after the entire batch succeeds.
        with OUTPUT_FILE.open("a", encoding="utf-8") as output_file:
            for result_record in batch_output:
                output_file.write(
                    json.dumps(
                        result_record,
                        ensure_ascii=False,
                    ) + "\n"
                )

        completed += len(batch_output)

        print(
            f"Saved {completed}/{total} test cases "
            f"to {OUTPUT_FILE}"
        )

    except Exception as exc:
        print(
            f"\nBatch starting at test case {start + 1} failed."
        )
        print(f"{type(exc).__name__}: {exc}")
        print(
            f"Previously completed results remain in "
            f"{OUTPUT_FILE.resolve()}"
        )
        raise

print("\nEvaluation completed successfully.")
print(f"Total evaluated: {completed}")
print(f"Output file: {OUTPUT_FILE.resolve()}")
