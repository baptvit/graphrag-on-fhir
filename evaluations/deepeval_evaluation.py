import os
from pathlib import Path
import pandas as pd

from langchain_google_vertexai import ChatVertexAI
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
BASE_DIR = Path("/home/baptvit/Documents/mestrado/master-experiments/evaluations/data")
SILVER_DIR = BASE_DIR / "silver"
GOLD_DIR = BASE_DIR / "gold"
SILVER_FILE = SILVER_DIR / f"{CONSUMER_ID}.csv"
GOLD_FILE = GOLD_DIR / f"{CONSUMER_ID}_deepeval.csv"


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


def create_test_cases(df: pd.DataFrame) -> list:
    """
    Create a list of LLMTestCase instances from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the silver data.

    Returns:
        list: List of LLMTestCase instances.
    """
    test_cases = []
    for _, row in df.iterrows():
        context = row.get("records", "No data available for this question.")
        if pd.isna(context):
            context = "No data available for this question."
        test_case = LLMTestCase(
            name=row.get("experiment_id", "Unnamed Test"),
            input=row.get("input", ""),
            actual_output=row.get("output", ""),
            retrieval_context=[context],
            context=[context],
        )
        test_cases.append(test_case)
    return test_cases


def run_evaluation(test_cases: list, model) -> dict:
    """
    Run evaluation metrics on the test cases.

    Args:
        test_cases (list): List of LLMTestCase instances.
        model: The LLM model to use for evaluation.

    Returns:
        dict: Dictionary containing the evaluation results.
    """
    dataset = EvaluationDataset(test_cases=test_cases)

    # Initialize metrics
    answer_relevancy_metric = AnswerRelevancyMetric(model=model)
    hallucination_metric = HallucinationMetric(threshold=THRESHOLD, model=model)
    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=THRESHOLD, model=model
    )

    # Run evaluation
    metrics = evaluate(
        dataset,
        [hallucination_metric, answer_relevancy_metric, contextual_relevancy_metric],
        ignore_errors=True,
        skip_on_missing_params=True,
    )
    return metrics.dict()


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


def main():
    """
    Main function to run the evaluation process.
    """
    # Ensure output directory exists
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize custom LLM model
    custom_llm = ChatVertexAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        request_parallelism=REQUEST_PARALLELISM,
    )

    # Load silver data
    try:
        df_silver = load_silver_data(SILVER_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    # Create test cases
    test_cases = create_test_cases(df_silver)

    # Run evaluation
    metrics_dict = run_evaluation(test_cases, custom_llm)

    # Process metrics
    df_gold = process_metrics(metrics_dict)

    # Add consumer ID
    df_gold["consumer_id"] = CONSUMER_ID

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
    df_final = df_gold[columns_order]

    # Save to CSV
    df_final.to_csv(GOLD_FILE, index=False)
    print(f"Evaluation results saved to {GOLD_FILE}")


if __name__ == "__main__":
    main()
