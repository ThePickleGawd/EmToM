from types import SimpleNamespace

from omegaconf import OmegaConf

from habitat_llm.llm.openai_chat import OpenAIChat


def _make_conf(model: str):
    return OmegaConf.create(
        {
            "verbose": False,
            "system_message": "You are an expert at task planning.",
            "keep_message_history": False,
            "generation_params": {
                "model": model,
                "max_tokens": 250,
                "temperature": 1,
                "top_p": 1,
                "stream": False,
                "stop": "Assigned!",
            },
        }
    )


def test_gpt_54_uses_responses_api_with_reasoning_budget(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            def chat_create(**kw):
                calls["chat"] = kw
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="chat response"))]
                )

            def responses_create(**kw):
                calls["responses"] = kw
                return SimpleNamespace(output_text="ok")

            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=chat_create)
            )
            self.responses = SimpleNamespace(create=responses_create)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("habitat_llm.llm.openai_chat.OpenAI", FakeOpenAI)

    llm = OpenAIChat(_make_conf("gpt-5.4"))
    out = llm.generate("plan this")

    assert out == "ok"
    assert "chat" not in calls
    assert calls["responses"]["model"] == "gpt-5.4"
    assert calls["responses"]["reasoning"] == {"effort": "medium"}
    assert calls["responses"]["max_output_tokens"] == 2048
    assert calls["responses"]["instructions"] == "You are an expert at task planning."


def test_non_reasoning_models_keep_250_token_chat_budget(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            def chat_create(**kw):
                calls["chat"] = kw
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

            def responses_create(**kw):
                calls["responses"] = kw
                return SimpleNamespace(output_text="responses response")

            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=chat_create)
            )
            self.responses = SimpleNamespace(create=responses_create)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("habitat_llm.llm.openai_chat.OpenAI", FakeOpenAI)

    llm = OpenAIChat(_make_conf("gpt-5.2"))
    out = llm.generate("plan this")

    assert out == "ok"
    assert "responses" not in calls
    assert calls["chat"]["model"] == "gpt-5.2"
    assert calls["chat"]["max_tokens"] == 250
    assert calls["chat"]["stop"] == "Assigned!"
