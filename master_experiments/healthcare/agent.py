from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
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

    # Define the tools and bind based on strategy
    tools = select_retrieval_strategy(strategy_name)

    llm_with_tools = llm.bind_tools(tools)

    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", HEALTHCARE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "history": lambda x: x.get("history", []),
        }
        | prompt
        | llm
        | OpenAIToolsAgentOutputParser(
            auto_call_tool_output=True
        )  # Auto-loopback for tool output
    )

    # Return AgentExecutor
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )


# response = chain.invoke(
#                 input={"input": str(user_input), "chat_history": [{}]},
#                 config={"configurable": {"session_id": "input_test"}},
#             )

#             utils.display_msg(response["output"], "assistant")
# response["output"], "assistant"
