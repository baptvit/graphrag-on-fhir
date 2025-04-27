# Analysis Steps

This documentation provides instructions on how to perform analysis shown in the paper, including summarization view and so on.

## Analysis Results

Below is a summary table of the evaluation status for different participants across various models and evaluation methods:

| Name  | Gemini - Microsoft Eval | Gemini - DeepEval | GPT-4o - Microsoft Eval | GPT-4o - DeepEval | Claude - Microsoft Eval | Claude - DeepEval |
| :----------------------------------- | :---------------------: | :---------------: | :---------------------: | :---------------: | :---------------------: | :---------------: |
| Allen322_Ferry570_ad1345...c625      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Beatris270_Bogan287_5b3645...c58     | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Edythe31_Morar593_9c3df3...23d       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Jacklyn830_Veum823_e0e1f2...a32      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Milton509_Ortiz186_d66b54...939      | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

✅ - Analysis completed  
❌ - Analysis pending

## Evaluation Models

The evaluations are performed using the following models:

- **GPT-4o** models include:
  - `gpt-4o`
  - `gpt-4o-2024-11-20`
  - `gpt-4o-2024-05-13`

- **Claude** models (collectively referred to as 'claudes') include:
  - `anthropic.claude-v3-haiku`
  - `anthropic.claude-v3-5-sonnet`
  - `anthropic.claude-v3-5-sonnet-v1`
  - `anthropic.claude-v3-5-sonnet-v2`
  - `anthropic.claude-v3-sonnet`

## Evaluation Process

Follow these steps to create consolidated log files and run the Microsoft evaluation.

### 1. Create Consolidated Log Files

To prepare the data for evaluation, consolidate log files for each participant (consumer ID).

#### Steps:

1. **Start Jupyter Lab**:
   Run the following command to start Jupyter Lab:

   ```bash
   uv run jupyter lab
   ```

2. **Open the Notebook**:
   - Navigate to and open the `create-silver-file-per-consumer-id` notebook within Jupyter Lab.

3. **Execute the Notebook**:
   - Pass the `consumer_id` to the `run()` function in the notebook.
   - Example:
     ```python
     run(consumer_id='YourConsumerID')
     ```
   - This will process the logs for the specified consumer ID.

4. **Locate the Generated Files**:
   - After running the notebook, a CSV file will be created in the `data/silver` directory.
   - The file will be named as `<consumer_id>-<model_name>.csv`.

### 2. Run the Microsoft Evaluation

Once the consolidated log files are prepared, proceed to run the evaluation using the Microsoft Eval script.

#### Steps:

1. **Navigate to the Evaluation Script**:
   - Open the `evaluations/microsoft_evaluation.py` script in your code editor.

2. **Configure Global Parameters**:
   - Edit the global parameters at the top of the script as needed.
   - Ensure that:
     - The file paths point to the correct directories.
     - The model names correspond to the data you've prepared.
     - Any additional configuration matches your environment.

3. **Run the Evaluation Script**:
   - Execute the script using the following command:

     ```bash
     uv run python -m evaluations.microsoft_evaluation
     ```

   - The script will process the data and output the evaluation results.

### 3. Run the DeepEval Evaluation

Once the consolidated log files are prepared, proceed to run the evaluation using the DeepEval script.

#### Steps:

1. **Navigate to the Evaluation Script**:
   - Open the `evaluations/deepeval_evaluation_incremental.py` script in your code editor.

2. **Configure Global Parameters**:
   - Edit the global parameters at the top of the script as needed.
   - Ensure that:
     - The file paths point to the correct directories.
     - The model names correspond to the data you've prepared.
     - Any additional configuration matches your environment.

3. **Run the Evaluation Script**:
   - Execute the script using the following command:

     ```bash
     uv run python -m evaluations.deepeval_evaluation_incremental
     ```

   - The script will process the data and output the evaluation results.

<!-- ## Additional Information

- **Dependencies**:
  - Ensure all required dependencies are installed.
  - You might need to install additional packages specified in `requirements.txt`.

- **Environment Configuration**:
  - Verify that your environment variables are set correctly, especially if the scripts rely on API keys or specific configurations.

- **Troubleshooting**:
  - If you encounter issues, check:
    - The logs generated during script execution.
    - That all file paths and model names are correct.
    - Your environment setup and configurations.

- **Evaluation Metrics**:
  - For detailed information on the evaluation metrics and methodology, refer to the documentation provided in the `evaluations` directory or consult the project's wiki.

- **Support**:
  - If you need assistance, please reach out to the project maintainers or open an issue in the repository. -->
