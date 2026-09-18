"""Query API routes (REST and SSE streaming)."""

import json
from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse
from verigraph.api.schemas import QueryRequest, QueryResponse
from verigraph.engine.pipeline import VeriGraphPipeline


router = APIRouter(prefix="/query", tags=["Query Engine"])


def get_pipeline(request: Request) -> VeriGraphPipeline:
    return request.app.state.pipeline


@router.post("", response_model=QueryResponse)
def execute_query(
    body: QueryRequest,
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Executes a synchronous verified query returning full provenance and NLI assertions."""
    result = pipeline.query(
        query_text=body.query,
        top_k=body.top_k,
        enable_graph=body.enable_graph,
    )
    return QueryResponse(
        query=result.query,
        response=result.response,
        strategy=result.strategy,
        faithfulness_score=result.faithfulness_score,
        answer_relevance_score=result.answer_relevance_score,
        latency_ms=result.latency_ms,
        retrieved_chunks=result.retrieved_chunks,
        claims=result.claims,
        subgraph=result.subgraph,
    )


@router.post("/stream")
async def execute_query_stream(
    body: QueryRequest,
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
):
    """Executes a streaming query yielding token events, graph subgraphs, and verification tags."""
    async def event_generator():
        async for item in pipeline.query_stream(query_text=body.query, top_k=body.top_k):
            yield {
                "event": item["event"],
                "data": json.dumps(item["data"]),
            }

    return EventSourceResponse(event_generator())
