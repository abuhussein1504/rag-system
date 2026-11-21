from fastapi import APIRouter

from src.api.dependencies import EmbedderDep, ProjectId, StoreDep
from src.api.schemas.search import SearchHit, SearchRequest, SearchResponse
from src.core.indexing.indexer import collection_name

search_router = APIRouter(prefix="/api/search", tags=["search"])


@search_router.post("/{project_id}", response_model=SearchResponse)
def search(
    project_id: ProjectId,
    body: SearchRequest,
    embedder: EmbedderDep,
    store: StoreDep,
) -> SearchResponse:
    vector = embedder.embed_query(body.query)
    hits = store.search(collection_name(project_id), vector, body.top_k)
    return SearchResponse(
        hits=[
            SearchHit(
                chunk_id=h.chunk.id,
                text=h.chunk.text,
                score=h.score,
                source=h.chunk.metadata.get("source", ""),
                page=h.chunk.metadata.get("page"),
            )
            for h in hits
        ]
    )

