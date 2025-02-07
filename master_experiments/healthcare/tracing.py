import datetime
import json
import os

import config


def write_json_to_file(json_data, experiment_step):
    # Construct the file path
    directory_path = os.path.join(
        "logs",
        f"{config.CONSUMER_ID}/{config.LLM_MODEL}/{config.EXPERTIMENT_STRATEGY}/{config.EXPERIMENT_ID}",
    )

    file_path = os.path.join(directory_path, f"{experiment_step}.json")

    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)

    try:
        # Write JSON data to file
        with open(file_path, "a") as file:
            json.dump(
                {
                    "experiment_id": [config.EXPERIMENT_ID],
                    "timestamp": [
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ],
                }
                | json_data,
                file,
                indent=4,
            )
        print(f"JSON data successfully written to {file_path}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {file_path}. Error: {e}")
