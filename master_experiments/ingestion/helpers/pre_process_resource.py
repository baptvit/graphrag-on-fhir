from langchain.globals import set_verbose
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import config
from master_experiments.prompts.preprocess_prompt import FHIR_PREPROCESS_PROMPT

set_verbose(True)


def preprocess_fhir_resouce(resource: str) -> str:
    """Pre process the input FHIR resource into a humam text format"""
    set_verbose(True)
    # llm = AzureChatOpenAI(
    #     api_key="",
    #     azure_deployment="gpt-4o-mini-2024-07-18",
    #     api_version="2023-08-01-preview",
    #     azure_endpoint="",
    # )
    llm = config.LLM_MODEL_INGESTION

    response = llm.invoke(
        #     f"""
        # As an expert in FHIR R4 conversion, Convert the following FHIR R4 resource into a human-readable text:  \"\"\"{resource}\"\"\". Ensure that all essential details are preserved and that the output does not include any binary or base64 data. The output should be plain text without any formatting.
        # Only provide the final text as the output:
        # """
        FHIR_PREPROCESS_PROMPT.format(resource=resource)
    )
    return str(response.content).replace('"', "").replace("'", "").replace("`", "")


def preprocess_fhir_resouce_embedding(fhir_text: str) -> str:
    """Pre process the input FHIR resource into a humam text format"""
    set_verbose(True)
    # embeddings = AzureOpenAIEmbeddings(
    #     model="text-embedding-ada-002",
    #     azure_deployment="text-embedding-ada-002",
    #     api_key="",
    #     api_version="2023-08-01-preview",
    #     azure_endpoint="",
    # )

    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    return embeddings.embed_query(fhir_text)
