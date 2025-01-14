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
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
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

## llm = ChatVertexAI(model="gemini-1.5-pro-002", temperature=0)

## llm = ChatOpenAI(
# model="meta/llama-3.1-405b-instruct-maas",
# base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
# #base_url=f"https://${ENDPOINT}/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions?",
# api_key=GOOGLE_TOKEN,
# )
# llm = AzureChatOpenAI(
#         api_key=AZURE_OPENAI_API_KEY,
#         azure_deployment="gpt-4o-mini-2024-07-18",
#         api_version=OPENAI_API_VERSION,
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     )

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
EXPERTIMENT_STRATEGY = "similarity_search_0_hop"  # similarity_search_1_hop

# Tokens limitis for enter in the MapReduce Summarizationborn
MAX_TOKENS = 1000000
MAX_CARACHTERES_IN_LLM_CONTEXT = 300000
CHUNK_SIZE = 22600
CHUNK_OVERLAP = 100

# Similarity Search configs on the information extraction tool
SIMILARITY_THRESHOLD = 0.84
K = 100000  # no restriction pratical

# User
CONSUMER_ID = "Jacklyn830_Veum823_e0e1f21a-22a7-d166-7bb1-63f6bbce1a32"

INPUT_QUESTION = {}

# Unique ID
EXPERIMENT_ID = ""

# CHAT History
CHAT_HISTORY = ""
