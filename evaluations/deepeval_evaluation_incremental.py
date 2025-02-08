import os
import config
import instructor

import pandas as pd

from pathlib import Path
from pydantic import BaseModel

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval import evaluate

# Configuration Constants
CONSUMER_ID = "Allen322_Ferry570_ad134528-56a5-35fd-c37f-466ff119c625"
MODEL_NAME = "gemini-1.5-pro"
TEMPERATURE = 0
REQUEST_PARALLELISM = 1
THRESHOLD = 0.5

# Directory Paths
SILVER_DIR = Path("/home/baptvit/repositories/graphrag-on-fhir/evaluations/data/silver")
GOLD_DIR = Path("/home/baptvit/repositories/graphrag-on-fhir/evaluations/data/gold")

SILVER_FILE = Path("/home/baptvit/repositories/graphrag-on-fhir/evaluations/data/silver/Allen322_Ferry570_ad134528-56a5-35fd-c37f-466ff119c625-gpt-4o-2024-08-06.csv")
GOLD_FILE = Path("/home/baptvit/repositories/graphrag-on-fhir/evaluations/data/gold/Allen322_Ferry570_ad134528-56a5-35fd-c37f-466ff119c625-gpt-4o-2024-08-06_deepeval.csv")


from deepeval.models import DeepEvalBaseLLM


## Azure OpenAI
class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"


## Google GCP
# class CustomAzure(DeepEvalBaseLLM):
#     def __init__(self):
#         self.model = config.LLM_MODEL_EVALUATION

#         # self.model = ChatVertexAI(model="gemini-1.5-pro", temperature=0, request_parallelism=1)

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
#         client = self.load_model()
#         instructor_client = instructor.from_gemini(
#             client=client,
#             mode=instructor.Mode.GEMINI_JSON,
#         )
#         resp = instructor_client.messages.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             response_model=schema,
#         )
#         return resp

#     async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
#         return self.generate(prompt, schema)

#     def get_model_name(self):
#         return config.LLM_MODEL_EVAL


def load_silver_data(file_path: Path) -> pd.DataFrame:
    """
    Load the silver dataset from a CSV file.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the silver data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Silver data file not found: {file_path}")
    return pd.read_csv(file_path)


def initialize_model():
    """
    Initialize the custom LLM model.

    Returns:
        ChatVertexAI: The initialized LLM model.
    """
    # custom_llm = ChatVertexAI(
    #     model=MODEL_NAME,
    #     temperature=TEMPERATURE,
    #     request_parallelism=REQUEST_PARALLELISM
    # )
    custom_llm = AzureOpenAI(model=config.LLM_MODEL_EVALUATION)
    return custom_llm


def initialize_metrics(model):
    """
    Initialize the evaluation metrics.

    Args:
        model: The LLM model to use for evaluation.

    Returns:
        list: List of metric instances.
    """
    answer_relevancy_metric = AnswerRelevancyMetric(model=model)
    hallucination_metric = HallucinationMetric(threshold=THRESHOLD, model=model)
    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=THRESHOLD, model=model
    )
    return [hallucination_metric, answer_relevancy_metric, contextual_relevancy_metric]


def load_existing_results(file_path: Path) -> pd.DataFrame:
    """
    Load existing results from the gold CSV file to support incremental execution.

    Args:
        file_path (Path): Path to the gold CSV file.

    Returns:
        pd.DataFrame: DataFrame containing existing results.
    """
    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()


def process_single_test_case(
    test_case: LLMTestCase, metrics_list: list, model
) -> pd.DataFrame:
    """
    Process a single test case and return the results as a DataFrame.

    Args:
        test_case (LLMTestCase): The test case to process.
        metrics_list (list): List of metrics to evaluate.
        model: The LLM model to use for evaluation.

    Returns:
        pd.DataFrame: DataFrame containing the evaluation results for the test case.
    """
    dataset = EvaluationDataset(test_cases=[test_case])

    # Run evaluation
    try:
        metrics = evaluate(
            dataset,
            metrics_list,
            ignore_errors=True,
            skip_on_missing_params=True,
        )
        metrics_dict = metrics.dict()
        df_results = process_metrics(metrics_dict)
        return df_results
    except Exception as e:
        print(f"Error processing test case {test_case.name}: {e}")
        # Create a DataFrame with error information
        error_data = {
            "test_name": test_case.name,
            "success": False,
            "metric_name": None,
            "threshold": None,
            "metric_success": False,
            "score": None,
            "reason": str(e),
            "strict_mode": None,
            "evaluation_model": None,
            "error": str(e),
            "evaluation_cost": None,
            "verbose_logs": None,
            "conversational": test_case.conversational,
            "multimodal": test_case.multimodal,
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "context": test_case.context[0] if test_case.context else None,
            "retrieval_context": (
                test_case.retrieval_context[0] if test_case.retrieval_context else None
            ),
        }
        df_error = pd.DataFrame([error_data])
        return df_error


def process_metrics(metrics_dict: dict) -> pd.DataFrame:
    """
    Process the metrics dictionary into a DataFrame.

    Args:
        metrics_dict (dict): Dictionary containing the evaluation results.

    Returns:
        pd.DataFrame: DataFrame containing the processed metrics.
    """
    rows = []
    for result in metrics_dict.get("test_results", []):
        for metric in result.get("metrics_data", []):
            row = {
                "test_name": result.get("name", ""),
                "success": result.get("success", False),
                "metric_name": metric.get("name", ""),
                "threshold": metric.get("threshold", ""),
                "metric_success": metric.get("success", False),
                "score": metric.get("score", 0.0),
                "reason": metric.get("reason", ""),
                "strict_mode": metric.get("strict_mode", False),
                "evaluation_model": metric.get("evaluation_model", ""),
                "error": metric.get("error", ""),
                "evaluation_cost": metric.get("evaluation_cost", 0.0),
                "verbose_logs": metric.get("verbose_logs", ""),
                "conversational": result.get("conversational", False),
                "multimodal": result.get("multimodal", False),
                "input": result.get("input", ""),
                "actual_output": result.get("actual_output", ""),
                "expected_output": result.get("expected_output", ""),
                "context": result.get("context", [None])[0],
                "retrieval_context": result.get("retrieval_context", [None])[0],
            }
            rows.append(row)
    df_metrics = pd.DataFrame(rows)
    return df_metrics


def save_results(df: pd.DataFrame, file_path: Path):
    """
    Save the results DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the evaluation results.
        file_path (Path): Path to the CSV file.
    """
    if file_path.exists():
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)


def main():
    """
    Main function to run the evaluation process.
    """
    # Ensure output directory exists
    # GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize custom LLM model
    custom_llm = initialize_model()

    # Initialize metrics
    metrics_list = initialize_metrics(custom_llm)

    # Load silver data
    try:
        df_silver = load_silver_data(SILVER_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    # Load existing results for incremental execution
    df_existing = load_existing_results(GOLD_FILE)
    processed_test_names = (
        set(df_existing["test_name"].unique()) if not df_existing.empty else set()
    )

    # Process test cases individually
    for index, row in df_silver.iterrows():
        test_name = row.get("experiment_id", f"Test_{index}")
        if test_name in processed_test_names:
            print(f"Skipping already processed test case: {test_name}")
            continue

        context = row.get("records", "No data available for this question.")
        if pd.isna(context) or context == "":
            context = "No data available for this question."

        test_case = LLMTestCase(
            name=test_name,
            input=row.get("input", ""),
            actual_output=row.get("output", ""),
            retrieval_context=[context],
            context=[context],
        )

        # Process the single test case
        df_result = process_single_test_case(test_case, metrics_list, custom_llm)

        # Add consumer ID
        df_result["consumer_id"] = CONSUMER_ID

        # Reorder columns
        columns_order = [
            "consumer_id",
            "test_name",
            "success",
            "metric_name",
            "threshold",
            "metric_success",
            "score",
            "reason",
            "strict_mode",
            "evaluation_model",
            "error",
            "evaluation_cost",
            "verbose_logs",
            "conversational",
            "multimodal",
            "input",
            "actual_output",
            "expected_output",
            "context",
            "retrieval_context",
        ]
        df_result = df_result[columns_order]

        # Save results incrementally
        save_results(df_result, GOLD_FILE)
        print(f"Processed and saved results for test case: {test_name}")

    print(f"All evaluations completed. Results saved to {GOLD_FILE}")


if __name__ == "__main__":
    main()
