from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

import config
from master_experiments.healthcare.tracing import write_json_to_file
from master_experiments.prompts.hallucination import HALLUCINATION_SELF_CHECK_PROMPT
from master_experiments.searchers.neo4j import (
    LexicalSearch0HopStrategy,
    LexicalSearch1HopStrategy,
    LexicalSearch2HopStrategy,
    Neo4jResourceSearcher,
    SearchStrategy,
    SimilaritySearch0HopStrategy,
    SimilaritySearch1HopStrategy,
    SimilaritySearch2HopStrategy,
)

from .save_outputs import read_string_from_file
from .summarization import count_tokens


class ResourceSearchTool:
    def __init__(
        self,
        strategy: SearchStrategy,
        resource_type: str = "",
        main_keys: str = "",
        user_query: str = "",
        llm_token_limit=config.MAX_TOKENS,
    ):
        self.strategy = strategy
        self.resource_type = resource_type
        self.main_keys = main_keys
        self.user_query = user_query
        self.llm_token_limit = llm_token_limit
        self.pass_map_reduce = False

    def execute_search(self):
        if self.resource_type:
            results = Neo4jResourceSearcher(self.strategy).search(
                resource_type=self.resource_type
            )
            records = "\n".join(results)
        else:
            searcher = Neo4jResourceSearcher(self.strategy)
            documents = searcher.search(self.main_keys)
            records = ""
            for document in documents:
                records += f"{document[0].page_content}\n"

        records_reduce = records
        if count_tokens(records) >= self.llm_token_limit:
            # TODO: Add the native approach to summarize the text
            records_reduce = records[:(config.MAX_CARACHTERES_IN_LLM_CONTEXT - 95000)]
            self.pass_map_reduce = True

        write_json_to_file(
            {
                "records": [records],
                "caracteres_count": [len(records)],
                "caracteres_count_after_reduce": [len(records_reduce)],
                "pass_map_reduce": [self.pass_map_reduce],
                "resource_type": [self.resource_type],
                "main_keys": [self.main_keys],
                "user_query": [self.user_query],
                "llm_token_limit": [self.llm_token_limit],
                "strategy_name": [self.strategy.__class__.__name__],
            },
            "tool_step",
        )

        if self.pass_map_reduce:
            return records_reduce

        return records


@tool
def lexical_search_0_hop(resource_type: str) -> str:
    """
    Return the patient’s electronic health record (EHR)

    Args:
        resource_type: Valid possible values in FHIR R4 resources type: i.e. Patient, Practitioner, PractitionerRole, Organization, Encounter, Observation, Condition, Procedure, Medication, MedicationRequest, Immunization, DiagnosticReport, AllergyIntolerance, CarePlan, CareTeam, Appointment, Coverage, Claim, Device, DocumentReference
    """
    tool = ResourceSearchTool(
        LexicalSearch0HopStrategy(), resource_type=resource_type, user_query=""
    )
    return tool.execute_search()


@tool
def lexical_search_1_hop(resource_type: str) -> str:
    """
    Return the patient’s electronic health record (EHR)

    Args:
        resource_type: Valid possible values in FHIR R4 resources type: i.e. Patient, Practitioner, PractitionerRole, Organization, Encounter, Observation, Condition, Procedure, Medication, MedicationRequest, Immunization, DiagnosticReport, AllergyIntolerance, CarePlan, CareTeam, Appointment, Coverage, Claim, Device, DocumentReference
    """
    tool = ResourceSearchTool(
        LexicalSearch1HopStrategy(), resource_type=resource_type, user_query=""
    )
    return tool.execute_search()


@tool
def lexical_search_2_hop(resource_type: str) -> str:
    """
    Return the patient’s electronic health record (EHR)

    Args:
        resource_type: Valid possible values in FHIR R4 resources type: i.e. Patient, Practitioner, PractitionerRole, Organization, Encounter, Observation, Condition, Procedure, Medication, MedicationRequest, Immunization, DiagnosticReport, AllergyIntolerance, CarePlan, CareTeam, Appointment, Coverage, Claim, Device, DocumentReference
    """
    tool = ResourceSearchTool(
        LexicalSearch2HopStrategy(), resource_type=resource_type, user_query=""
    )
    return tool.execute_search()


@tool
def similarity_search_0_hop(main_keys: str) -> str:
    """
    Return the patient’s electronic health record (EHR)

    Args:
        main_keys (str): A string containing key terms or clinical concepts extracted from the user’s input. These may include:
                    Medications (e.g., medications).
                    Allergies (e.g., allergys).
                    Conditions (e.g., conditions, observations).
                    Lab Results (e.g., blood tests, lab results).
                    Other Clinical Concepts (e.g., treatment plans or surgeries).
    """
    tool = ResourceSearchTool(
        SimilaritySearch0HopStrategy(), main_keys=main_keys, user_query=""
    )
    return tool.execute_search()


@tool
def similarity_search_1_hop(main_keys: str) -> str:
    """
    Return the patient’s electronic health record (EHR)

    Args:
    main_keys (str): A string containing key terms or clinical concepts extracted from the user’s input. These may include:
                    Medications (e.g., medications).
                    Allergies (e.g., allergys).
                    Conditions (e.g., conditions, observations).
                    Lab Results (e.g., blood tests, lab results).
                    Other Clinical Concepts (e.g., treatment plans or surgeries).
    """
    tool = ResourceSearchTool(
        SimilaritySearch1HopStrategy(), main_keys=main_keys, user_query=""
    )
    return tool.execute_search()


@tool
def similarity_search_2_hop(main_keys: str) -> str:
    """
    Return the patient’s electronic health record (EHR)

    Args:
    main_keys (str): A string containing key terms or clinical concepts extracted from the user’s input. These may include:
                    Medications (e.g., medications).
                    Allergies (e.g., allergys).
                    Conditions (e.g., conditions, observations).
                    Lab Results (e.g., blood tests, lab results).
                    Other Clinical Concepts (e.g., treatment plans or surgeries).
    """
    tool = ResourceSearchTool(
        SimilaritySearch2HopStrategy(), main_keys=main_keys, user_query=""
    )
    return tool.execute_search()


@tool
def self_check_hallucination(
    previous_actual_output: str, previous_user_query: str
) -> str:
    """
    Evaluates whether the previous output is hallucinated based on the provided context.

    Args:
        previous_actual_output (str): the previus output from the modal.
        previous_user_query (str): the previus user query to be answer.

    Returns:
        str: A message indicating whether the previous output is a hallucination and the reason for the assessment.
    """
    llm = AzureChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        azure_deployment="gpt-4o-2024-08-06",
        api_version=config.API_VERSION,
        azure_endpoint=config.OPENAI_ENDPOINT,
    )

    final_prompt = HALLUCINATION_SELF_CHECK_PROMPT.format(
        user_question=previous_user_query,
        model_output=previous_actual_output,
        context=read_string_from_file("./last_context.txt"),
    )

    response = llm.invoke(final_prompt)
    return response.content


@tool
def context_enricher(query: str) -> str:
    """
    Enriches the user question with relevant context.
        user query: input query
    """
    return "'The Medication Request resource details a prescription for Acetaminophen 160 MG Chewable Tablet, which has been marked as stopped. This order was categorized under community medication and was authored on November 23, 2016, by Dr. Alvin56 Crona259. The patient associated with this request has a unique identifier, and the request is linked to a specific encounter. The dosage instruction advises the patient to take the medication as needed.' -> [('authored_on', '11/23/2016'), ('encounter', 'The encounter with ID cf60427f-6701-abf2-82fc-945837df2152 took place at Holy Family Hospital, where the patient, Beatris270 Bogan287, was seen for an encounter related to symptoms of otitis media. This encounter occurred on November 23, 2016, starting at 1:37 PM and concluding at 1:52 PM. Dr. Alvin56 Crona259 served as the primary performer during this visit. The encounter is classified as an ambulatory visit and is officially documented under the identifier cf60427f-6701-abf2-82fc-945837df2152.'), ('prescription', 'The claim, identified by the ID 9cab8190-7c02-ebf0-e3d4-dbb918c4f395, is currently active and pertains to a pharmacy service. It was created on November 23, 2016, and covers a billable period from 1:37 PM to 1:52 PM on the same day. The claim is associated with a patient referenced by the UUID 5b3645de-a2d0-d016-0839-bab3757c4c58 and is linked to a prescription identified by the UUID 6ac1a7c9-300a-a7cf-f140-524b291660c7. The provider for this claim is HOLY FAMILY HOSPITAL, which is referenced by an organization identifier. The claim is prioritized as normal and is primarily covered by Medicare insurance. The claim includes one item, which is Acetaminophen 160 MG Chewable Tablets, with a total claim amount of $91.78. The encounter related to this item is referenced by the UUID cf60427f-6701-abf2-82fc-945837df2152.'), ]\n'The Medication Request resource outlines a prescription that has been stopped for a patient, identified by the reference urn:uuid:5b3645de-a2d0-d016-0839-bab3757c4c58. This order, categorized as a community medication request, was authored on February 26, 2022, by Dr. Alvin56 Crona259. The medication specified in this request is Penicillin V Potassium 250 MG Oral Tablet, which is associated with the treatment of a streptococcal sore throat. The request was made during an encounter referenced by urn:uuid:fc6a22f2-848e-d29e-0a8b-df2e931ae617.' -> [('reason_reference_0', 'The patient, identified by the reference urn:uuid:5b3645de-a2d0-d016-0839-bab3757c4c58, was diagnosed with Streptococcal sore throat, confirmed as the clinical condition on February 17, 2021. This condition was categorized as an encounter diagnosis and was resolved by February 26, 2021. The diagnosis was recorded on the same date as the onset, which marks the beginning of the clinical status. The patients encounter related to this diagnosis is referenced by urn:uuid:4e7837d4-083f-9360-e1e3-3550ea57d9d9.'), ('authored_on', '02/26/2022'), ('encounter', 'On February 26, 2022, Beatris270 Bogan287 was involved in an encounter at Holy Family Hospital, which lasted from 5:37 PM to 5:52 PM. This encounter, classified as an ambulatory visit, was conducted to address symptoms related to a streptococcal sore throat. Dr. Alvin56 Crona259 served as the primary performer during this encounter. The encounter has been marked as finished and is officially identified by the value fc6a22f2-848e-d29e-0a8b-df2e931ae617.'), ('prescription', 'The active claim, identified by the ID d55fbee6-1a4f-07a1-e231-3d510a2261d5, was created on February 26, 2022, and pertains to a pharmacy service. It is associated with a patient whose unique identifier is 5b3645de-a2d0-d016-0839-bab3757c4c58. The claim covers a billable period from February 26, 2022, at 5:37 PM to February 26, 2022, at 5:52 PM. The provider of the service is Holy Family Hospital, referenced by the organization identifier. The priority of the claim is classified as normal. It includes a prescription for Penicillin V Potassium 250 MG Oral Tablet, which is documented under the UMLS code 834061. The claim is supported by Medicare insurance, which is the primary coverage. The total amount billed for the service is $324.32 USD.'), ]\n'The medication request with the identifier 3cf09561-5d36-5abd-61dd-7596bea2c5ac is currently active and is categorized under community orders. It was authored on August 22, 2017, by Dr. Alvin56 Crona259. The request is for Fexofenadine hydrochloride 30 MG Oral Tablet, which the patient is instructed to take as needed. The patient associated with this request is identified by a unique reference, and the request is linked to a specific encounter as well.' -> [('authored_on', '08/22/2017'), ('encounter', 'On August 22, 2017, Beatris270 Bogan287 had an encounter at Holy Family Hospital, which lasted from 1:37 PM to 1:52 PM. This encounter, classified as an ambulatory visit, was focused on addressing symptoms related to perennial allergic rhinitis. Dr. Alvin56 Crona259 served as the primary performer during this visit. The official identifier for this encounter is 9fc6351d-7e69-20c6-c597-856f8e70483d, and it has been marked as finished.'), ('prescription', 'The active claim identified by the ID 30deaebb-5574-1a5f-8ed0-fc99deffc423 was created on August 22, 2017. This claim is categorized as a pharmacy claim for a patient referenced by the UUID 5b3645de-a2d0-d016-0839-bab3757c4c58. The billable period for this claim spans from August 22, 2017, at 13:37 to August 22, 2017, at 13:52. The healthcare provider associated with this claim is Holy Family Hospital, which is identified by the organization reference. The claim has a normal priority and is linked to a prescription referenced by the UUID 3cf09561-5d36-5abd-61dd-7596bea2c5ac. The insurance coverage for this claim is provided by Medicare, making it the focal coverage. The claim includes one item: Fexofenadine hydrochloride 30 MG Oral Tablet, which was provided during an encounter referenced by the UUID 9fc6351d-7e69-20c6-c597-856f8e70483d. The total amount for this claim is $455.32 USD.'), ]\n'The MedicationRequest resource documents a prescription for Amoxicillin 500 MG Oral Tablet, which has been stopped. The prescription was initiated on November 23, 2016, by Dr. Alvin56 Crona259 during a community encounter. The patient, identified by a specific UUID, was instructed to take the medication at regular intervals, completing the prescribed course unless directed otherwise. The dosage was specified as one tablet, to be taken three times a day. This request is categorized under community medication requests, reflecting its context of use in a non-hospital setting.' -> [('authored_on', '11/23/2016'), ('encounter', 'The encounter with ID cf60427f-6701-abf2-82fc-945837df2152 took place at Holy Family Hospital, where the patient, Beatris270 Bogan287, was seen for an encounter related to symptoms of otitis media. This encounter occurred on November 23, 2016, starting at 1:37 PM and concluding at 1:52 PM. Dr. Alvin56 Crona259 served as the primary performer during this visit. The encounter is classified as an ambulatory visit and is officially documented under the identifier cf60427f-6701-abf2-82fc-945837df2152.'), ('prescription', 'The claim, identified by the ID aaa2b803-6ed4-cf5e-c1c4-1ecca768ac14, is currently active and pertains to a pharmacy service. It was created on November 23, 2016, and includes a billable period from 1:37 PM to 1:52 PM on the same day. The patient referenced in this claim is associated with a UUID of 5b3645de-a2d0-d016-0839-bab3757c4c58, while the provider is Holy Family Hospital, identified through a specific organizational reference. The claim is categorized with a normal processing priority and is linked to a prescription identified by the UUID 6948c88e-d15c-3f11-a635-be707cb15632. The insurance coverage for this claim is provided by Medicare, which is the primary insurance. There is one item listed on the claim, which is Amoxicillin 500 MG Oral Tablet, recognized by the code 308192 in the RxNorm database, and it is associated with an encounter identified by the UUID cf60427f-6701-abf2-82fc-945837df2152. The total amount charged for this service is 103.33 USD.'), ]\n"


def select_retrieval_strategy(strategy_name: str):
    """Select the appropriate tools for a retrieval strategy."""
    strategy_tools = {
        "lexical_search_0_hop": [lexical_search_0_hop],
        "lexical_search_1_hop": [lexical_search_1_hop],
        "lexical_search_2_hop": [lexical_search_2_hop],
        "similarity_search_0_hop": [similarity_search_0_hop],
        "similarity_search_1_hop": [similarity_search_1_hop],
        "similarity_search_2_hop": [similarity_search_2_hop],
        "any": [
            lexical_search_1_hop,
            similarity_search_1_hop,
            self_check_hallucination,
        ],
        "app": [
            lexical_search_0_hop,
            similarity_search_0_hop,
            self_check_hallucination,
        ],
    }

    return strategy_tools.get(strategy_name, strategy_tools["any"])
