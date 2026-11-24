from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.dependencies import (
    EmbedderDep,
    LLMDep,
    ProjectId,
    SettingsDep,
    StoreDep,
    get_reranker,
)
from src.api.schemas.ask import AskCitation, AskRequest, AskResponse
from src.core.generation.generator import AnswerGenerator
from src.core.llm.base import LLMError
from src.core.retrieval.factory import build_retriever
from src.core.indexing.indexer import collection_name
from src.models.enums import RetrieverName

ask_router = APIRouter(prefix="/api/ask", tags=["ask"])


@ask_router.post("/{project_id}", response_model=AskResponse)
def ask(
    project_id: ProjectId,
    body: AskRequest,
    settings: SettingsDep,
    embedder: EmbedderDep,
    store: StoreDep,
    llm: LLMDep,
) -> AskResponse:

    if store.count(collection_name(project_id)) == 0:
        raise HTTPException(
            status_code=404, detail=f"Project '{project_id}' has no indexed documents"
        )
    
    name = body.retriever or settings.DEFAULT_RETRIEVER
    retriever = build_retriever(
        name,
        project=project_id,
        processed_dir=Path(settings.DATA_DIR) / "processed" / project_id,
        embedder=embedder,
        store=store,
        reranker=get_reranker() if name == RetrieverName.HYBRID_RERANK else None,
    )
    try:
        answer = AnswerGenerator(retriever, llm).answer(body.question, body.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"Language model error: {exc}") from None
    return AskResponse(
        answer=answer.text,
        refused=answer.refused,
        citations=[AskCitation(**c.model_dump()) for c in answer.citations],
    )

