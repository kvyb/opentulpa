from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from opentulpa.agent.lc_messages import AIMessage, HumanMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


class _State(TypedDict):
    messages: Annotated[list[Any], add_messages]


@pytest.mark.asyncio
async def test_astream_model_surfaces_provider_chunks_to_langgraph(tmp_path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    model = FakeListChatModel(responses=["abc"])

    async def agent_node(state: _State) -> dict[str, list[Any]]:
        response = await runtime.astream_model(
            model,
            list(state["messages"]),
            model_name="z-ai/glm-5.1",
            call_context={"call_site": "graph_agent"},
        )
        return {"messages": [response]}

    builder = StateGraph(_State)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    chunks: list[str] = []
    async for message_chunk, metadata in graph.astream(
        {"messages": [HumanMessage(content="hi")]},
        stream_mode="messages",
    ):
        assert metadata.get("langgraph_node") == "agent"
        chunks.append(str(getattr(message_chunk, "content", "")))

    assert chunks == ["a", "b", "c"]


class _StreamingToolCallModel:
    async def astream(self, _messages: list[Any]):
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "fake_tool", "args": "{", "id": "call_1", "index": 0}
            ],
        )
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": None, "args": '"step":1}', "id": None, "index": 0}
            ],
        )


class _ConfigCapturingStreamModel:
    def __init__(self) -> None:
        self.config_seen: Any | None = None

    async def astream(self, _messages: list[Any], config: Any | None = None):
        self.config_seen = config
        yield AIMessageChunk(content="hello")


@pytest.mark.asyncio
async def test_astream_model_preserves_streamed_tool_calls(tmp_path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )

    response = await runtime.astream_model(
        _StreamingToolCallModel(),
        [HumanMessage(content="use a tool")],
        model_name="z-ai/glm-5.1",
        call_context={"call_site": "graph_agent"},
    )

    assert isinstance(response, AIMessage)
    assert response.tool_calls == [
        {"name": "fake_tool", "args": {"step": 1}, "id": "call_1", "type": "tool_call"}
    ]


@pytest.mark.asyncio
async def test_astream_model_forwards_graph_stream_config(tmp_path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    model = _ConfigCapturingStreamModel()
    stream_config = {"callbacks": ["graph-callback-marker"]}

    response = await runtime.astream_model(
        model,
        [HumanMessage(content="hi")],
        model_name="z-ai/glm-5.1",
        call_context={"call_site": "graph_agent"},
        stream_config=stream_config,
    )

    assert model.config_seen is stream_config
    assert isinstance(response, AIMessage)
    assert response.content == "hello"
