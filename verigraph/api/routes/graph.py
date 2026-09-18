"""Knowledge Graph exploration routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from verigraph.api.schemas import GraphStatsResponse, PathSearchRequest
from verigraph.core.models import Subgraph
from verigraph.engine.pipeline import VeriGraphPipeline


router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


def get_pipeline(request: Request) -> VeriGraphPipeline:
    return request.app.state.pipeline


@router.get("/stats", response_model=GraphStatsResponse)
def get_graph_stats(
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
) -> GraphStatsResponse:
    """Returns knowledge graph topological metrics and PageRank centrality."""
    stats = pipeline.graph_engine.get_stats()
    central = pipeline.graph_engine.get_central_entities(top_k=10)
    return GraphStatsResponse(
        total_nodes=stats["total_nodes"],
        total_edges=stats["total_edges"],
        density=stats["density"],
        connected_components=stats["connected_components"],
        central_entities=central,
    )


@router.get("/subgraph", response_model=Subgraph)
def get_subgraph(
    max_nodes: int = Query(default=35, ge=5, le=100),
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
) -> Subgraph:
    """Returns the primary connected knowledge subgraph for interactive visualization."""
    central = pipeline.graph_engine.get_central_entities(top_k=8)
    seed_ids = [c["canonical_id"] for c in central] if central else []

    if not seed_ids and pipeline.graph_engine.graph.number_of_nodes() > 0:
        seed_ids = list(pipeline.graph_engine.graph.nodes)[:5]

    return pipeline.graph_engine.get_subgraph_for_entities(
        entity_ids=seed_ids,
        max_depth=2,
        max_nodes=max_nodes,
    )


@router.post("/path")
def find_relational_path(
    body: PathSearchRequest,
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
):
    """Finds the shortest relational path between two entities across document boundaries."""
    path = pipeline.graph_engine.find_relational_path(body.source_entity, body.target_entity)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No relational path found between '{body.source_entity}' and '{body.target_entity}'",
        )
    return {"source": body.source_entity, "target": body.target_entity, "hops": path}
