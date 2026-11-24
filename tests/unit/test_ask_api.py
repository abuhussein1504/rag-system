from main import app
from src.api.dependencies import get_llm
from src.core.generation.prompt import REFUSAL_TEXT
from src.core.llm.base import LLMError


class ScriptedLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error

    def generate(self, system, prompt):
        if self.error:
            raise self.error
        return self.reply


def index_project(client, project="p1"):
    client.post(
        f"/api/data/upload/{project}", files={"file": ("a.txt", b"cats purr softly. " * 40)}
    )
    client.post(f"/api/data/process/{project}")
    assert client.post(f"/api/data/index/{project}").json()["chunks_indexed"] > 0


def ask(client, project, question="What do cats do?"):
    return client.post(f"/api/ask/{project}", json={"question": question})


def test_ask_returns_an_answer_with_a_clean_source_name(client):
    index_project(client)
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM("Cats purr [1].")
    response = ask(client, "p1")

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["answer"] == "Cats purr [1]."
    assert body["citations"][0]["source"] == "a.txt"  # no upload prefix


def test_ask_passes_a_model_refusal_through(client):
    index_project(client)
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM(REFUSAL_TEXT)
    body = ask(client, "p1").json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_ask_unknown_project_is_404_not_a_fake_refusal(client):
    index_project(client)  # project "p1" exists, "project_p1" does not
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM("unused")
    assert ask(client, "project_p1").status_code == 404
    assert ask(client, "nope").status_code == 404


def test_ask_turns_a_model_failure_into_502(client):
    index_project(client)
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM(error=LLMError("quota"))
    assert ask(client, "p1").status_code == 502


def test_ask_rejects_an_empty_question(client):
    index_project(client)
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM("unused")
    assert client.post("/api/ask/p1", json={"question": ""}).status_code == 422

