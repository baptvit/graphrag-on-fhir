## Prompt used in the ingestion phase where we preprocess the json resource into a text string
# FHIR_PREPROCESS_PROMPT = '''
#     As an expert in FHIR R4 conversion, Convert the following FHIR R4 resource into a detailed, human-readable text. Ensure that all details are preserved and that the output does not include any binary or base64 data. The output MUST be plain text without any formatting. Leave out ids, uuids, links and external links. Only provide the final text as the output:

#     Take the FHIR R4 resource: """{resource}"""

#     Do not make up any information. Only provide the information that is present in the FHIR R4 resource.
#     '''


FHIR_PREPROCESS_PROMPT = '''
    As an expert in FHIR R4 conversion, Convert the following FHIR R4 resource into a detailed, human-readable text. Ensure that all details are preserved and that the output does not include any binary, base64 data, ids and uuids.

    Take the FHIR R4 resource: """{resource}"""
    Do not make up any information. Only provide the information that is present in the FHIR R4 resource. 

    The format MUST BE as follow property: value,
    The property and value must be humam understandable
    Include Medical number and Coding Classification System
    Only provide the final text as the output
    '''
