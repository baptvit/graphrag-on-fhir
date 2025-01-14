## Prompt used for the Map phase of the MapReduce summarization of long context
SUMMARIZATION_MAP_PROMPT = """As a detailed and comprehensive summary expert, summarize the following health records. 
            Highlight key information including costs, places, doctors, and any other relevant details, base on the data: '''{text}''' 
            Do not add anything new, just summarize what is present on the context."""


# SUMMARIZATION_REDUCE_PROMPT = """
#             "Combine the following summaries into a cohesive summary:\n{text}
#             """
