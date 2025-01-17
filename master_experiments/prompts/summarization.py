## Prompt used for the Map phase of the MapReduce summarization of long context
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
