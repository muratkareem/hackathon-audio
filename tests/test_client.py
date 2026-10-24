import json
import httpx
import pytest
from core.schemas import Answer
from core.llm.client import LocalLLM, InferenceError


def install_transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_structured_output_retry(monkeypatch, settings):
    calls = []

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": settings.llm_model}]})
        body = json.loads(request.content)
        assert body["format"]["additionalProperties"] is False
        calls.append(body)
        content = (
            "invalid"
            if len(calls) == 1
            else json.dumps(
                {"answer": "Known", "references": [0], "insufficient_evidence": False}
            )
        )
        return httpx.Response(200, json={"message": {"content": content}})

    install_transport(monkeypatch, handler)
    assert LocalLLM(settings).generate(Answer, "system", "input").answer == "Known"
    assert len(calls) == 2


def test_missing_model(monkeypatch, settings):
    install_transport(
        monkeypatch, lambda request: httpx.Response(200, json={"models": []})
    )
    with pytest.raises(InferenceError, match="ollama_model_missing"):
        LocalLLM(settings).generate(Answer, "system", "input")


def test_connection_failure(monkeypatch, settings):
    def handler(request):
        raise httpx.ConnectError("private host path", request=request)

    install_transport(monkeypatch, handler)
    with pytest.raises(InferenceError, match="ollama_unavailable"):
        LocalLLM(settings).models()


def test_invalid_output_bounded(monkeypatch, settings):
    calls = []

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": settings.llm_model}]})
        calls.append(1)
        return httpx.Response(200, json={"message": {"content": "invalid"}})

    install_transport(monkeypatch, handler)
    with pytest.raises(InferenceError, match="invalid_model_output"):
        LocalLLM(settings).generate(Answer, "system", "input")
    assert len(calls) == 2
