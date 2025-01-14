import tiktoken
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

import config
from master_experiments.prompts.summarization import SUMMARIZATION_MAP_PROMPT


# Initialize the AzureChatOpenAI client
def get_openai_client():
    # openai_client = AzureChatOpenAI(
    #     api_key=config.OPENAI_API_KEY,
    #     azure_deployment="gpt-4o-mini-2024-07-18",
    #     api_version=config.API_VERSION,
    #     azure_endpoint=config.OPENAI_ENDPOINT,
    # )
    openai_client = config.LLM_MODEL_SUMMARIZATION
    return openai_client


# Map function: Generate summaries for text chunks
def map_summarize(
    text_chunks,
    openai_client,
):
    summarized_chunks = []
    print("Number of chuncks, ", len(text_chunks))
    for chunk in text_chunks:
        prompt = SUMMARIZATION_MAP_PROMPT.format(chunk=chunk)
        response = openai_client.invoke(prompt)
        summarized_chunks.append(response.content)
        # Work around to avoid 419 status code
    return summarized_chunks


# Reduce function: Concatenate and summarize the summarized chunks
def reduce_summarize(
    summarized_chunks,
):
    return "\n".join(summarized_chunks)


# Main MapReduce summarization function
def map_reduce_summarize_naive(long_text: str, user_query: str):
    # Initialize the OpenAI client
    openai_client = get_openai_client()

    # Split the long text into manageable chunks
    # text_splitter = CharacterTextSplitter(chunk_size=15000, chunk_overlap=100, separators=["\n\n"])
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    text_chunks = text_splitter.split_text(long_text)

    # Apply the map function
    summarized_chunks = map_summarize(text_chunks, openai_client)

    # Apply the reduce function
    final_summary = reduce_summarize(summarized_chunks)
    return final_summary


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


def map_recude_native_summarization(large_text: str):
    # Step 2: Set up an LLM (e.g., OpenAI's GPT model)
    large_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of achieving its goals.

    Some popular applications of AI include expert systems, natural language processing (NLP), speech recognition, and machine vision.
    """

    llm = config.LLM_MODEL_SUMMARIZATION

    # Step 3: Define the prompts for Map and Reduce stages
    map_prompt = PromptTemplate(
        input_variables=["text"], template=SUMMARIZATION_MAP_PROMPT
    )
    reduce_prompt = PromptTemplate(
        input_variables=["text"],
        template="Combine the following summaries into a cohesive summary:\n{text}",
    )

    # Step 4: Set up a Text Splitter to divide the text into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=100, chunk_overlap=20
    )
    text_chunks = text_splitter.split_text(large_text)

    # Step 5: Create the MapReduceChain
    # map_reduce_chain = MapReduceChain(
    #     llm=llm,
    #     map_prompt=map_prompt,
    #     reduce_prompt=reduce_prompt,
    #     input_key="text"  # The input key expected by the chain
    # )

    map_reduce_chain = MapReduceChain(
        llm=llm,
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        text_splitter=text_splitter,
    )

    # Step 6: Run the chain with the input data
    result = map_reduce_chain.run({"text": text_chunks})

    # Output the result
    print("Final Summary:")
    print(result)
    import ipdb

    ipdb.set_trace()
    return " ".join(result)


if __name__ == "__main__":
    map_recude_native_summarization("test")
