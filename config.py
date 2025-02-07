import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_openai import AzureChatOpenAI

load_dotenv(".env")

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
# GCP Configurations
# LLM_MODEL = "gemini-1.5-pro"
# llm = ChatVertexAI(model=LLM_MODEL, temperature=0, safety_settings=safety_settings)

# Azure Configurations
#LLM_MODEL = "gpt-4o-2024-08-06"
# LLM_MODEL = "anthropic.claude-v3-opus"
LLM_MODEL = "gpt-4o-2024-11-20"
llm = AzureChatOpenAI(
    api_key=OPENAI_API_KEY,
    azure_deployment=LLM_MODEL,
    api_version=API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
)

LLM_MODEL_INGESTION = llm

LLM_MODEL_GENERATION = llm

LLM_MODEL_SUMMARIZATION = llm


# Azure Configurations
#LLM_MODEL_EVAL = "gpt-4o-2024-11-20"  # gpt-4o, gpt-4o-2024-11-20, gpt-4o-2024-05-13
LLM_MODEL_EVAL = "anthropic.claude-v3-5-sonnet-v2" # anthropic.claude-v3-haiku, anthropic.claude-v3-5-sonnet, anthropic.claude-v3-5-sonnet-v2, anthropic.claude-v3-5-sonnet-v1, anthropic.claude-v3-sonnet
llm_eval = AzureChatOpenAI(
    api_key=OPENAI_API_KEY,
    azure_deployment=LLM_MODEL_EVAL,
    api_version=API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
)
LLM_MODEL_EVALUATION = llm_eval


# EXPERTIMENT_STRATEGY = "lexical_search_0_hop" # lexical_search_0_hop, lexical_search_1_hop, lexical_search_2_hop, lexical_search_3_hop
EXPERTIMENT_STRATEGY = "similarity_search_0_hop"  # similarity_search_1_hop

# Tokens limitis for enter in the MapReduce Summarizationborn
MAX_TOKENS = 128_000
# MAX_TOKENS = 200_000
MAX_CARACHTERES_IN_LLM_CONTEXT = MAX_TOKENS * 4

# Similarity Search configs on the information extraction tool
SIMILARITY_THRESHOLD = 0.84
K = 100000  # no restriction pratical

# User
CONSUMER_ID = "Allen322_Ferry570_ad134528-56a5-35fd-c37f-466ff119c625"

INPUT_QUESTION = {}

# Unique ID
EXPERIMENT_ID = ""

# CHAT History
CHAT_HISTORY = ""
