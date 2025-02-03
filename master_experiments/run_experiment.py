import time
from pathlib import Path
from typing import Any, Callable, List, Union

import pydantic
from dotenv import load_dotenv
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

import config
from master_experiments.healthcare.agent_v1 import create_healthcare_agent_executor
from master_experiments.healthcare.tracing import write_json_to_file
from master_experiments.prompts.system_prompt import HEALTHCARE_SYSTEM_PROMPT


class Input(pydantic.BaseModel):
    input: str = pydantic.Field(
        ...,
        max_length=5000,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "input"}},
    )
    # The field extra defines a chat widget.
    # Please see documentation about widgets in the main README.
    # The widget is used in the playground.
    # Keep in mind that playground support for agents is not great at the moment.
    # To get a better experience, you'll need to customize the streaming output
    # for now.
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage, ToolMessage]] = (
        pydantic.Field(
            ...,
            extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
        )
    )


class Output(pydantic.BaseModel):
    output: Any


def _is_valid_identifier(value: str) -> bool:
    """Check if the session ID is in a valid format."""
    # Use a regular expression to match the allowed characters
    return True


def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a session ID factory that creates session IDs from a base dir.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A session ID factory that creates session IDs from a base path.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        """Get a chat history from a session ID."""
        if not _is_valid_identifier(session_id):
            return FileChatMessageHistory("")
        file_path = base_dir_ / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path))

    return get_chat_history


def setup_chain(strategy):
    return (
        RunnableWithMessageHistory(
            create_healthcare_agent_executor(strategy),
            create_session_factory("chat_histories"),
            input_messages_key="input",
            history_messages_key="history",
        )
        .with_types(input_type=Input, output_type=Output)
        .with_config({"run_name": "agent"})
    )


if __name__ == "__main__":
    # run experiment here
    load_dotenv(".env")

    # EXPERIMENT_QUESTIONS = {
    # "Q1": "Whats my current medication and how should I take them ?",
    # "Q2": "What are the most common side effects for each medication I am taking?"
    # }

    EXPERIMENT_QUESTIONS = {
        "Q1": "What's my current medications and how should I be taking them ?",
        "Q2": "What are my documented allergies, and how severe are they ?",
        "Q3": "Can you summarize my current medical conditions ?",
        "Q4": "What are my recent laboratory values, what do they mean, and how can I improve them ?",
        "Q5": "Can you summarize my care plan history ?",
        "Q6": "Can you provide a breakdown of my medical bills ?",
        "Q7": "What procedures have I undergone recently, and what were the outcomes ?",
        "Q8": "Can you summarize my immunization history ?",
    }

    list_search_strategies = [
        "lexical_search_0_hop",
        "lexical_search_1_hop",
        # "lexical_search_2_hop",
        "similarity_search_0_hop",
        "similarity_search_1_hop",
        # "similarity_search_2_hop",
    ]

    # LOOP SEARCH STRATEGIES
    for strategy in list_search_strategies:
        config.EXPERTIMENT_STRATEGY = strategy
        # LOOP QUESTIONS
        for key in EXPERIMENT_QUESTIONS.keys():
            question = EXPERIMENT_QUESTIONS.get(key)
            config.INPUT_QUESTION.update({key: question})
            config.EXPERIMENT_ID = (
                key + ":" + config.LLM_MODEL + ":" + config.EXPERTIMENT_STRATEGY
            )
            config.CHAT_HISTORY = (
                config.CONSUMER_ID
                + ":"
                + config.LLM_MODEL
                + ":"
                + config.EXPERTIMENT_STRATEGY
            )
            INPUT_QUESTION = question
            time.sleep(2)
            with get_openai_callback() as cb:
                chain = setup_chain(strategy)
                start_time = time.time()
                response = chain.invoke(
                    input={"input": str(INPUT_QUESTION), "chat_history": []},
                    config={"configurable": {"session_id": config.CHAT_HISTORY}},
                )
                total_time = time.time() - start_time

                write_json_to_file(
                    {
                        "full_response": [str(response)],
                        "system_promt": [HEALTHCARE_SYSTEM_PROMPT],
                        "input": [str(INPUT_QUESTION)],
                        "output": [response["output"]],
                    },
                    "output_step",
                )
                write_json_to_file(
                    {
                        "total_cost": [cb.total_cost],
                        "total_tokens": [cb.total_tokens],
                        "successful_requests": [cb.successful_requests],
                        "completion_tokens": [cb.completion_tokens],
                        "prompt_tokens": [cb.prompt_tokens],
                        "total_time": [total_time],
                    },
                    "tokens_step",
                )
