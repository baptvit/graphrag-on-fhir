import os
from pathlib import Path

from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory,
)

## Neo4j Configurations
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://192.168.2.129:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# OPEN Keys
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("OPENAI_API_VERSION", "2023-08-01-preview")

# Google
ENDPOINT = "us-central1-aiplatform.googleapis.com"
REGION = "us-central1"
PROJECT_ID = ""
GOOGLE_TOKEN = ""

# Ingestion data
SYNTHEA_DATA_DIR = BASE_DIR = (
    Path(__file__).resolve().parent / "fhir_data" / "stanford_llm_on_fhir"
)

# MODELS
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# EXPERIMENTS CONFIGURATIONS
LLM_MODEL = "gemini-1.5-pro"


LLM_MODEL_INGESTION = ChatVertexAI(
    model=LLM_MODEL, temperature=0, safety_settings=safety_settings
)

LLM_MODEL_GENERATION = ChatVertexAI(
    model=LLM_MODEL, temperature=0, safety_settings=safety_settings
)

LLM_MODEL_SUMMARIZATION = ChatVertexAI(
    model=LLM_MODEL, temperature=0, safety_settings=safety_settings
)

LLM_MODEL_EVALUATION = ChatVertexAI(
    model=LLM_MODEL, temperature=0, safety_settings=safety_settings
)


# EXPERTIMENT_STRATEGY = "lexical_search_0_hop" # lexical_search_0_hop, lexical_search_1_hop, lexical_search_2_hop, lexical_search_3_hop
EXPERTIMENT_STRATEGY = "app"  # similarity_search_1_hop

# Tokens limitis for enter in the MapReduce Summarizationborn
MAX_TOKENS = 1_000_000
MAX_CARACHTERES_IN_LLM_CONTEXT = MAX_TOKENS * 4

# Similarity Search configs on the information extraction tool
SIMILARITY_THRESHOLD = 0.85
K = 20  # no restriction pratical

# User
CONSUMER_ID = "Beatris270_Bogan287_5b3645de-a2d0-d016-0839-bab3757c4c58"

INPUT_QUESTION = {}

# Unique ID
EXPERIMENT_ID = ""

# CHAT History
CHAT_HISTORY = ""
