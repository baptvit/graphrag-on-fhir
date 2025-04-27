# GraphRAG on Electronic Health Record: A Knowledge Graph-Enhanced RAG Approach for Healthcare Information Access
The increasing importance of digital health and the need for better health literacy require effective methods to access and understand Electronic Health Records (EHRs). While Large Language Models (LLMs) show promise in this domain, traditional Retrieval-Augmented Generation (RAG) struggles to handle the complex, interconnected nature of clinical data. GraphRAG emerges as a powerful alternative, leveraging knowledge graphs (KGs) to capture semantic relationships within EHRs. This research evaluates the effectiveness of graph expansion in a GraphRAG to enhance information retrieval from FHIR-formatted medical data. We propose a 1-hop expansion approach built upon a lexical search baseline which, while inheriting some limitations of traditional keyword-based retrieval, significantly enhances LLMs' access to comprehensive and diverse contextual information. Our evaluation, using synthetic patient data and a targeted set of questions across five models, reveals that the 1-hop expansion strategy consistently outperforms the baseline in subjective metrics like comprehensiveness and diversity, and frequently in qualitative metrics such as answer and contextual relevancy. These results highlight the potential of our proposal to enhance LLM response when querying intricate medical data.

## Citation
This is a academic project so if you want to replicate or use any code. Just cite this work.

# Database
[Anonymous drive](https://drive.google.com/drive/folders/1wyRkhDf786SEQNHtR8ElQ6Q-qwgnX_gB)


# Installation

## Prerequisites
Python 3.11 or higher

UV for dependency management

## Steps
1. Follow the [official UV installation guide.](https://docs.astral.sh/uv/getting-started/installation/)

2. To install all dependencies
```bash
uv sync
```

3. Activate the enviroment just created

```bash
source .venv/bin/activate
```

# Configuration Documentation

The project uses a config.py file to manage all configurations. Below is a detailed breakdown of the available settings:

| **Category**               | **Variable**                          | **Description**                                                                 | **Default Value/Environment Variable**                     |
|----------------------------|---------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------|
| **Neo4j Configurations**    | `NEO4J_URI`                          | URI for connecting to the Neo4j database.                                      | `os.getenv("NEO4J_URI", "neo4j://192.168.2.129:7687")`     |
|                            | `NEO4J_USER`                         | Username for Neo4j authentication.                                             | `os.getenv("NEO4J_USERNAME", "neo4j")`                     |
|                            | `NEO4J_PASSWORD`                     | Password for Neo4j authentication.                                             | `"password"`                                               |
|                            | `NEO4J_DATABASE`                     | Name of the Neo4j database.                                                    | `os.getenv("NEO4J_DATABASE", "neo4j")`                     |
| **OpenAI Configurations**   | `OPENAI_API_KEY`                     | API key for OpenAI services.                                                   | `os.getenv("AZURE_OPENAI_API_KEY")`                        |
|                            | `OPENAI_ENDPOINT`                    | Endpoint for OpenAI services.                                                  | `os.getenv("AZURE_OPENAI_ENDPOINT")`                       |
|                            | `API_VERSION`                        | Version of the OpenAI API.                                                     | `os.getenv("OPENAI_API_VERSION", "2023-08-01-preview")`    |
| **Google Configurations**   | `ENDPOINT`                           | Endpoint for Google Vertex AI services.                                        | `"us-central1-aiplatform.googleapis.com"`                  |
|                            | `REGION`                             | Region for Google Vertex AI services.                                          | `"us-central1"`                                            |
|                            | `PROJECT_ID`                         | Google Cloud project ID.                                                       | `""`                                                       |
|                            | `GOOGLE_TOKEN`                       | Token for Google Cloud authentication.                                         | `""`                                                       |
| **Ingestion Data**          | `SYNTHEA_DATA_DIR`                   | Directory path for Synthea FHIR data.                                          | `Path(__file__).resolve().parent / "fhir_data" / "stanford_llm_on_fhir"` |
| **Model Configurations**    | `safety_settings`                    | Safety settings for Google Vertex AI models.                                   | `{HarmCategory: HarmBlockThreshold.BLOCK_NONE}`            |
|                            | `LLM_MODEL`                          | Name of the LLM model used.                                                    | `"gemini-1.5-pro"`                                         |
|                            | `LLM_MODEL_INGESTION`                | Vertex AI model for ingestion tasks.                                           | `ChatVertexAI(model=LLM_MODEL, temperature=0, safety_settings=safety_settings)` |
|                            | `LLM_MODEL_GENERATION`               | Vertex AI model for generation tasks.                                          | `ChatVertexAI(model=LLM_MODEL, temperature=0, safety_settings=safety_settings)` |
|                            | `LLM_MODEL_SUMMARIZATION`            | Vertex AI model for summarization tasks.                                       | `ChatVertexAI(model=LLM_MODEL, temperature=0, safety_settings=safety_settings)` |
|                            | `LLM_MODEL_EVALUATION`               | Vertex AI model for evaluation tasks.                                          | `ChatVertexAI(model=LLM_MODEL, temperature=0, safety_settings=safety_settings)` |
| **Experiment Configurations** | `EXPERTIMENT_STRATEGY`              | Strategy for experiments (e.g., similarity search).                            | `"similarity_search_0_hop"`                                |
|                            | `MAX_TOKENS`                         | Maximum tokens allowed for MapReduce summarization.                            | `1000000`                                                  |
|                            | `MAX_CARACHTERES_IN_LLM_CONTEXT`     | Maximum characters allowed in the LLM context.                                 | `300000`                                                   |
|                            | `CHUNK_SIZE`                         | Size of chunks for text processing.                                            | `22600`                                                    |
|                            | `CHUNK_OVERLAP`                      | Overlap between chunks for text processing.                                    | `100`                                                      |
|                            | `SIMILARITY_THRESHOLD`               | Threshold for similarity search in information extraction.                     | `0.84`                                                     |
|                            | `K`                                  | Maximum number of results for similarity search.                               | `100000`                                                   |
| **User Configurations**     | `CONSUMER_ID`                        | Unique identifier for the user/consumer.                                       | `"Beatris270_Bogan287_5b3645de-a2d0-d016-0839-bab3757c4c58"` |
|                            | `INPUT_QUESTION`                     | Placeholder for user input questions.                                          | `{}`                                                       |
| **Experiment Metadata**     | `EXPERIMENT_ID`                      | Unique identifier for the experiment.                                          | `""`                                                       |
|                            | `CHAT_HISTORY`                       | Placeholder for chat history.                                                  | `""`                                                       |

---

# Running the Project
## Run ingestion
1. Config the `LLM_MODEL_INGESTION` in the config.py and the Neo4j instance to ingest the data on.
2. Move the target files from `master_experiments/fhir_data/stanford_llm_on_fhir` to `master_experiments/fhir_data/data`.
Every bundle in the `master_experiments/fhir_data/data` will be convert in the Graph Representation.
3. Run the ingestion script
```bash
$ uv run python -m master_experiments.ingestion.ingestion_service_parallel
```
## Run APP
1. Run the langchain serve
```bash 
$ uv run langchain serve
```

## Run experiments
1. Config the paraments in the `config.py` and `master_experiments/run_experiment.py`, i.e the questions to be ask and the target consumer_id.
2. Run the experiment script
```bash 
$ uv run python -m master_experiments.run_experiment
```

# Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bugfix.

3. Submit a pull request with a detailed description of your changes.

# License
This project is licensed under the MIT License. See the LICENSE file for details.