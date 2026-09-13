import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import SecretStr

from harness_tools import make_delegating_tools
from oai_compat_chat import OAICompatChat

load_dotenv()

tools = make_delegating_tools(root_dir="/", virtual_mode=False, shell=True)  # + execute

tools_dict: Dict[str, BaseTool] = {t.name: t for t in tools}

llm = OAICompatChat(
    model="deepseek-flash",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=SecretStr(os.environ["DEEPSEEK_API_KEY"]),
    metal_log=True,
    preserve_reasoning_to_request=True,  # copy reasoning from the AIMessages passed in this call
)


llm_with_tools = llm.bind_tools(tools)

messages: List[BaseMessage] = [
    SystemMessage(
        content="You are a Helpful Assistant. You can call tools to answer questions. You are currently at /app."
    ),
    HumanMessage(
        content="Please autonomously improve the sphere packing program `initial_program.py` and evaluate using `evaluator.py`. You can make multiple snapshots, edit as you like, write experiment logs, etc. and install anything in pip e.g. optimizers, matplotlib, etc. Briefly report the numbers for each evaluation, and saved at which snapshot."
    )
]

for i in range(100):
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    logger.info("turn {}: response: {}", i, response.content)
    if len(response.tool_calls) == 0:
        break

    results: List[ToolMessage] = [tools_dict[c["name"]].invoke(c) for c in response.tool_calls]
    messages += results
