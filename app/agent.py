import os
import app.config as config

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from master_experiments.healthcare.tools import select_retrieval_strategy
from master_experiments.prompts.system_prompt import HEALTHCARE_SYSTEM_PROMPT_APP


def create_healthcare_agent_executor(strategy="app"):
    os.environ["GOOGLE_API_KEY"] = ""

    llm = config.LLM_MODEL_EVALUATION

    # Define the tools and bind based on strategy
    tools = select_retrieval_strategy(strategy)

    llm_with_tools = llm.bind_tools(tools)

    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                HEALTHCARE_SYSTEM_PROMPT_APP.format(
                    function_name="lexical_search_1_hop, similarity_search_1_hop and self_check_hallucination"
                ),
            ),
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
        | llm_with_tools
        | OpenAIToolsAgentOutputParser(
            auto_call_tool_output=True
        )  # Auto-loopback for tool output
    )

    # Return AgentExecutor
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
