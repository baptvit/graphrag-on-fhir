from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from master_experiments.healthcare.tools import select_retrieval_strategy
from master_experiments.prompts.system_prompt import HEALTHCARE_SYSTEM_PROMPT


def create_healthcare_agent_executor(strategy_name=config.EXPERTIMENT_STRATEGY):
    # TODO: Improve for future use more abstract all to pass the LLM
    # llm = AzureChatOpenAI(
    #     api_key=config.OPENAI_API_KEY,
    #     azure_deployment="gpt-4o-2024-08-06",
    #     api_version=config.API_VERSION,
    #     azure_endpoint=config.OPENAI_ENDPOINT,
    # )
    llm = config.LLM_MODEL_EVALUATION

    # ENDPOINT="us-central1-aiplatform.googleapis.com"
    # REGION="us-central1"
    # PROJECT_ID="master-experiments-project"

    # import vertexai

    # vertexai.init(project=PROJECT_ID, location=REGION)

    # import vertexai
    # from google.auth import default, transport

    # vertexai.init()
    # credentials, _ = default()
    # auth_request = transport.requests.Request()
    # credentials.refresh(auth_request)

    # MODEL_LOCATION = "us-central1"

    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(
    #     model="meta/llama-3.1-405b-instruct-maas",
    #     base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
    #     #base_url=f"https://${ENDPOINT}/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions?",
    #     api_key=credentials.token,
    # )

    # Define the tools and bind based on strategy
    tools = select_retrieval_strategy(strategy_name)

    llm_with_tools = llm.bind_tools(tools, tool_choice=strategy_name)

    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "system",
                HEALTHCARE_SYSTEM_PROMPT.format(
                    function_name=config.EXPERTIMENT_STRATEGY
                ),
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)

    # Return AgentExecutor
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=2)


# Keeping track of conversation history
# def update_chat_history(chat_history, input_message, response_message):
#     # chat_history.append({"role": "human", "content": input_message})
#     # chat_history.append({"role": "assistant", "content": response_message})
#     pass

# #agent = create_healthcare_agent_executor("similarity_search_1_hop")


# agent = create_healthcare_agent_executor(config.EXPERTIMENT_STRATEGY)

# # Example usage
# def main():
#     chat_history = []

#     user_input = "What's my current medications and how should I be taking them?"
#     response = agent.invoke({"input": user_input, "chat_history": []})

#     response_message = response["output"]
#     update_chat_history(chat_history, user_input, response_message)

#     import ipdb
#     ipdb.set_trace()

#     user_input = "What are my documented allergies, and how severe are they?"

#     response = agent.invoke({"input": user_input, "chat_history": chat_history})

#     response_message = response["output"]
#     update_chat_history(chat_history, user_input, response_message)

#     import ipdb
#     ipdb.set_trace()

#     user_input = "Can you summarize my current medical conditions?"

#     response = agent.invoke({"input": user_input, "chat_history": chat_history})

#     response_message = response["output"]
#     update_chat_history(chat_history, user_input, response_message)

#     import ipdb
#     ipdb.set_trace()

#     user_input = "What are my recent laboratory values, what do they mean, and how can I improve them?"

#     response = agent.invoke({"input": user_input, "chat_history": chat_history})

#     response_message = response["output"]
#     update_chat_history(chat_history, user_input, response_message)

#     import ipdb
#     ipdb.set_trace()

#     user_input = "Can you summarize my care plan history ?"

#     response = agent.invoke({"input": user_input, "chat_history": chat_history})

#     response_message = response["output"]
#     update_chat_history(chat_history, user_input, response_message)

#     import ipdb
#     ipdb.set_trace()

#     user_input = "Can you provide full summary of my medical bills ?"

#     response = agent.invoke({"input": user_input, "chat_history": chat_history})

#     response_message = response["output"]
#     update_chat_history(chat_history, user_input, response_message)

#     import ipdb
#     ipdb.set_trace()

#     user_input = "What procedures have I undergone recently, and what were the outcomes?"

#     response = agent.invoke({"input": user_input, "chat_history": chat_history})

#     response_message = response["output"]
#     update_chat_history(chat_history, user_input, response_message)


# if __name__ == "__main__":
#     main()
