import tiktoken

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

SUMMARIZATION_MAP_PROMPT = """As a detailed and comprehensive summary expert, your task is to summarize the following health records. Focus on extracting and highlighting the following key information:
1. **Costs**: Any mentioned expenses, fees, or financial details.
2. **Places**: Locations such as hospitals, clinics, or treatment centers.
3. **Doctors**: Names, specialties, or roles of healthcare providers.
4. **Relevant Details**: Key medical information, diagnoses, treatments, medications, or procedures.

**Instructions:**
- Be concise and factual. Only summarize information explicitly present in the context.
- Do not add, infer, or speculate any details not explicitly mentioned.
- Organize the summary in a clear and structured format.

**Health Records:**
'''{text}'''"""

SUMMARIZATION_REDUCE_PROMPT = """
    The following are summaries of different chunks of text from a patient's health records:
{text}

Your task is to combine these summaries into one coherent and personalized final summary. Focus on the following key aspects:
1. **Patient Information**: Include the patient's name, age, and any relevant personal details.
2. **Medical History**: Summarize diagnoses, treatments, medications, and procedures.
3. **Healthcare Providers**: Mention doctors, specialists, or clinics involved in the patient's care.
4. **Costs and Insurance**: Highlight any financial details, such as costs, payments, or insurance coverage.
5. **Timeline**: Organize the information chronologically or by relevance to provide a clear timeline of events.

**Instructions:**
- Be concise and factual. Only include information explicitly present in the summaries.
- Use clear and professional language suitable for healthcare documentation.
- Ensure the final summary is well-structured and easy to understand.
            """


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
    from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory,
    )

    safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

    llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0, safety_settings=safety_settings)

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
        token_max=128000,  # Adjust based on model token limits
    )

    # Map-reduce chain
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="text",
    )

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=128000/5,  # Adjust based on your needs
        chunk_overlap=200,  # Overlap to maintain context
    )
    docs = text_splitter.create_documents([long_text])

    # Run the map-reduce chain to generate the summary
    summary = map_reduce_chain.run(docs)
    return summary


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
