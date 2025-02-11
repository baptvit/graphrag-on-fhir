#!/usr/bin/env python
# coding: utf-8

import json
import re
import pandas as pd
import config

from pathlib import Path

from itertools import combinations

# Constants
CONSUMER_ID = "Allen322_Ferry570_ad134528-56a5-35fd-c37f-466ff119c625"
SOURCE_INTERMEDIATE_PATH = "/home/baptvit/repositories/graphrag-on-fhir/evaluations/data/silver/Allen322_Ferry570_ad134528-56a5-35fd-c37f-466ff119c625-llama-3-70b-instruct-awq.csv"
SOURCE_GOLD_PATH = "/home/baptvit/repositories/graphrag-on-fhir/evaluations/data/gold/Allen322_Ferry570_ad134528-56a5-35fd-c37f-466ff119c625-llama-3-70b-instruct-awq.csv_microsoft_eval.csv"

AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = ""
OPENAI_API_VERSION = ""

EVALUATION_MICROSOFT_PROMPT = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness** and **Diversity**.
---Goal---
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, **Empowerment** and **Directness**.
- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgements about the question? 
- **Directness**: How specifically and clearly does the answer address the question?
For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.
Here is the question:
{query}
Here are the two answers:
**Answer 1:**
{answer1}
**Answer 2:**
{answer2}
Evaluate both answers using the four criteria listed above and provide detailed explanations for each criterion.
Output your evaluation in the following MUST BE JSON VALID format:
{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Directness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
"""


def extract_winners_txt(text):
    # Define regex patterns to capture the winners for each section
    patterns = {
        "comprehensiveness": r"'Comprehensiveness':\s*{\s*'Winner':\s*'([^']+)'.*?}",
        "diversity": r"'Diversity':\s*{\s*'Winner':\s*'([^']+)'.*?}",
        "empowerment": r"'Empowerment':\s*{\s*'Winner':\s*'([^']+)'.*?}",
        "directness": r"'Directness':\s*{\s*'Winner':\s*'([^']+)'.*?}",
        "overall_winner": r"'Overall Winner':\s*{\s*'Winner':\s*'([^']+)'.*?}",
    }
    try:
        text = text.replace('"', "'")
        # Extract winners using regex
        winners = {
            key: re.search(pattern, text).group(1) for key, pattern in patterns.items()
        }

        # Create and return a DataFrame from the extracted data
        # df = pd.DataFrame([winners])
        return winners
    except Exception as e:
        print("Error extracting winners:", e)
        return {
            "comprehensiveness": "",
            "diversity": "",
            "empowerment": "",
            "directness": "",
            "overall_winner": "",
        }


def extract_winners_json(json):
    # Define regex patterns to capture the winners for each section
    try:
        return {
            "comprehensiveness": json["Comprehensiveness"]["Winner"],
            "diversity": json["Diversity"]["Winner"],
            "empowerment": json["Empowerment"]["Winner"],
            "directness": json["Directness"]["Winner"],
            "overall_winner": json["Overall Winner"]["Winner"],
        }
    except Exception as e:
        print("Error extracting winners:", e)
        return {
            "comprehensiveness": "",
            "diversity": "",
            "empowerment": "",
            "directness": "",
            "overall_winner": "",
        }


def cast_text_to_json(text: str) -> dict:
    try:
        # Remove unwanted characters like triple backticks, if present
        cleaned_text = text.strip("`").strip()

        # Find the start and end of the JSON segment
        json_start = cleaned_text.find("{")
        json_end = cleaned_text.rfind("}") + 1  # +1 to include the closing bracket

        if json_start == -1 or json_end == -1:
            raise json.JSONDecodeError("Invalid JSON format detected.")

        # Extract the JSON substring
        json_substring = cleaned_text[json_start:json_end]

        # Parse the JSON string into a Python dictionary
        json_data = json.loads(json_substring)

        return json_data

    except ValueError as ve:
        print("Error processing text:", ve)
        return text
    except json.JSONDecodeError as jde:
        print("Error decoding JSON:", jde)
        return text
    except Exception as e:
        print("An unexpected error occurred:", e)
        return text


class MicrosoftEvaluationProcessor:
    def __init__(self):
        self.consumer_id = CONSUMER_ID

        self.llm = config.LLM_MODEL_EVALUATION

        self.intermediate_df = pd.read_csv(SOURCE_INTERMEDIATE_PATH)

    def generate_pairs(self):
        """Generate pairs of experiments with equal input values."""
        pairs = []
        grouped = self.intermediate_df.groupby("input")
        for _, group in grouped:
            unique_ids = group["experiment_id"].unique()
            if (
                len(unique_ids) > 1
            ):  # Only generate pairs if there are at least two experiments
                pairs.extend(combinations(unique_ids, 2))
        return pairs

    def evaluate_pair(self, exp1_id, exp2_id):
        exp1_output = self.intermediate_df.loc[
            self.intermediate_df["experiment_id"] == exp1_id, "output"
        ].values[0]
        exp2_output = self.intermediate_df.loc[
            self.intermediate_df["experiment_id"] == exp2_id, "output"
        ].values[0]

        exp1_name = self.intermediate_df.loc[
            self.intermediate_df["experiment_id"] == exp1_id, "strategy_name"
        ].values[0]
        exp2_name = self.intermediate_df.loc[
            self.intermediate_df["experiment_id"] == exp2_id, "strategy_name"
        ].values[0]

        exp1_input = self.intermediate_df.loc[
            self.intermediate_df["experiment_id"] == exp1_id, "input"
        ].values[0]
        exp2_input = self.intermediate_df.loc[
            self.intermediate_df["experiment_id"] == exp2_id, "input"
        ].values[0]

        if exp1_input == exp2_input:
            print(f"Compare: {exp1_input} with {exp2_input}")
            print(f"Compare: {exp1_name} with {exp2_name}")
            print("----------------------------------------")
            prompt_eval = EVALUATION_MICROSOFT_PROMPT.format(
                query=exp1_input, answer1=exp1_output, answer2=exp2_output
            )

            response = self.llm.invoke(prompt_eval)

            try:
                output_parsed = cast_text_to_json(response.content)
                dict_winners = extract_winners_json(output_parsed)
            except Exception:
                output_parsed = response.content
                dict_winners = extract_winners_txt(output_parsed)

            return {
                "consumer_id": self.consumer_id,
                "experiment_1": exp1_id,
                "expriment_1_name": exp1_name,
                "experiment_2": exp2_id,
                "expriment_2_name": exp2_name,
                "experiment_1_output": exp1_output,
                "experiment_2_output": exp2_output,
                "model_evaluation": output_parsed,
                "response": response,
            } | dict_winners

    def process_evaluations(self):
        """Process evaluations and append results incrementally."""
        results = []
        for exp1, exp2 in self.generate_pairs():
            evaluation_result = self.evaluate_pair(exp1, exp2)
            if evaluation_result:
                results.append(evaluation_result)

                # Optionally, save incremental results to a file
                pd.DataFrame(results).to_csv(
                    SOURCE_GOLD_PATH.format(consumer_id=self.consumer_id), index=False
                )

        return pd.DataFrame(results)
    
    def is_new_experiment(self, exp1, exp2):
        if Path(SOURCE_GOLD_PATH).exists():
            df = pd.read_csv(SOURCE_GOLD_PATH)
            df_filter = df[(df["experiment_1"] == exp1) & (df["experiment_2"] == exp2) ]
            return df_filter.empty
        else:
            return True

    def save_results(self, df: pd.DataFrame, file_path: Path):
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
    
    def process_evaluations_incremental(self):
        """Process evaluations and write results directly to a CSV incrementally."""
        for exp1, exp2 in self.generate_pairs():
            
            if not self.is_new_experiment(exp1, exp2):
                print(f"Skinping already processed experiments pairs: {exp1} and {exp2}")
                continue 

            evaluation_result = self.evaluate_pair(exp1, exp2)
            if evaluation_result:
                df_result = pd.DataFrame([evaluation_result])

                # Append header only for the first row
                # df_result.to_csv(f, header=not header_written, index=False)
                # header_written = True
                self.save_results(df_result, Path(SOURCE_GOLD_PATH))

    # def save_results(self, df_results):
    #     df_results.to_csv(
    #         SOURCE_GOLD_PATH.format(consumer_id=self.consumer_id), index=False
    #     )


def main():
    processor = MicrosoftEvaluationProcessor()
    # Use process_evaluations_incremental for large datasets
    processor.process_evaluations_incremental()


if __name__ == "__main__":
    main()
