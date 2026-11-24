from typing import Any, Callable

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langsmith import Client, tracing_context
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider


class PassThrough(AgentMiddleware):
    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(request)


@pytest.mark.parametrize("langsmith_tracing", [False, True])
def test_llm_span_nests_under_model_node_with_wrap_model_call_middleware(
    langsmith_tracing: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    agent = create_agent(
        model=FakeListChatModel(responses=["hello"]),
        tools=[],
        middleware=[PassThrough()],
    )
    unreachable_client = Client(api_key="ls-test", api_url="http://127.0.0.1:9")
    with (
        tracing_context(enabled=langsmith_tracing, client=unreachable_client),
        tracer_provider.get_tracer(__name__).start_as_current_span("request"),
    ):
        agent.invoke({"messages": [HumanMessage("hi")]})

    spans = {s.name: s for s in in_memory_span_exporter.get_finished_spans()}
    assert spans["FakeListChatModel"].parent
    assert spans["model"].context
    assert spans["FakeListChatModel"].parent.span_id == spans["model"].context.span_id
