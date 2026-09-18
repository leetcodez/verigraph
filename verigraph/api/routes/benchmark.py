"""Evaluation and benchmarking routes."""

from pathlib import Path
from fastapi import APIRouter, Depends, Request
from verigraph.api.schemas import BenchmarkRunRequest
from verigraph.engine.pipeline import VeriGraphPipeline
from evaluation.evaluator import SystemEvaluator


router = APIRouter(prefix="/benchmark", tags=["Benchmarking"])


def get_pipeline(request: Request) -> VeriGraphPipeline:
    return request.app.state.pipeline


@router.post("/run")
def run_system_benchmark(
    body: BenchmarkRunRequest,
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
):
    """Triggers automated evaluation comparing Naive RAG vs Hybrid RAG vs VeriGraph."""
    dataset_path = Path(__file__).resolve().parent.parent.parent.parent / "evaluation" / "dataset.json"
    evaluator = SystemEvaluator(pipeline, dataset_path)
    results = evaluator.run_benchmark(num_samples=body.num_samples)
    return results
