import tiktoken
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import (
    CharacterTextSplitter,
)

import config
from master_experiments.prompts.summarization import (
    SUMMARIZATION_MAP_PROMPT,
    SUMMARIZATION_REDUCE_PROMPT,
)


def summarize_long_text(long_text: str) -> str:
    """
    Summarizes a long text using a map-reduce approach with LangChain.

    Args:
        long_text (str): The input text to summarize.
        openai_api_key (str): Your OpenAI API key.

    Returns:
        str: The final summarized text.
    """
    # Initialize the LLM
    llm = config.LLM_MODEL_SUMMARIZATION

    # Define the map and reduce prompts
    map_template = SUMMARIZATION_MAP_PROMPT
    map_prompt = PromptTemplate.from_template(map_template)

    reduce_template = SUMMARIZATION_REDUCE_PROMPT
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Create the map and reduce chains
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combine documents using StuffDocumentsChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="text"
    )

    # Reduce documents chain
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=config.MAX_TOKENS,  # Adjust based on model token limits
    )

    # Map-reduce chain
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="text",
    )

    target_size = round(config.MAX_CARACHTERES_IN_LLM_CONTEXT / 4)
    actual_size = len(long_text)

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=actual_size
        / ((actual_size / 4) / target_size),  # Adjust based on your needs
        chunk_overlap=100,  # Overlap to maintain context
    )

    #     text_splitter = TokenTextSplitter(
    #     chunk_size=config.MAX_TOKENS / 4,  # Maximum number of tokens per chunk
    #     chunk_overlap=100,  # Number of tokens to overlap between chunks
    #     encoding_name="cl100k_base",  # Tokenizer encoding (default for OpenAI models)
    # )

    docs = text_splitter.create_documents([long_text])
    # import ipdb
    # ipdb.set_trace()
    # Run the map-reduce chain to generate the summary
    summary = map_reduce_chain.run(docs)
    return summary


import concurrent.futures

from langchain.docstore.document import Document


def summarize_long_text(long_text: str) -> str:
    """
    Summarizes a long text using a map-reduce approach with LangChain and parallel execution.
    """
    # Initialize the LLM
    llm = config.LLM_MODEL_SUMMARIZATION

    # Define the map and reduce prompts
    map_template = SUMMARIZATION_MAP_PROMPT
    map_prompt = PromptTemplate.from_template(map_template)

    reduce_template = SUMMARIZATION_REDUCE_PROMPT
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Create the map and reduce chains
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combine documents using StuffDocumentsChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="text"
    )

    # Reduce documents chain
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=config.MAX_TOKENS,  # Adjust based on model token limits
    )

    target_size = round(config.MAX_CARACHTERES_IN_LLM_CONTEXT / 4)
    actual_size = len(long_text)
    chuck_size = actual_size / ((actual_size / 4) / target_size)
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=chuck_size,  # Adjust based on your model's token limit
        chunk_overlap=100,  # Overlap to maintain context
    )
    docs = text_splitter.create_documents([long_text])

    # Function to process a single document chunk
    def process_chunk(chunk):
        return map_chain.run([chunk])

    # Process chunks in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all chunks for parallel processing
        future_to_chunk = {
            executor.submit(process_chunk, chunk): chunk for chunk in docs
        }
        summaries = []
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                summary = future.result()
                summaries.append(summary)
            except Exception as e:
                print(f"Error processing chunk: {e}")

    # Combine the summaries into a single document
    combined_summary = " ".join(summaries)
    doc = Document(page_content=combined_summary, metadata={"source": "local"})
    return reduce_documents_chain.run([doc])


def count_tokens(json_string, model_name="gpt-4"):
    """
    Counts the number of tokens and characters in a JSON file.

    Args:
        json_file_path (str): The path to the JSON file.
        model_name (str): The model name for tokenization. Default is "gpt-3.5-turbo".

    Returns:
        dict: A dictionary containing the number of tokens and characters.
    """
    # Initialize tiktoken encoder
    try:
        encoder = tiktoken.encoding_for_model(model_name)
    except KeyError:
        raise ValueError(f"Model '{model_name}' not supported by tiktoken.")

    # Tokenize the JSON string
    tokens = encoder.encode(json_string)
    token_count = len(tokens)

    return token_count
