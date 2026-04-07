"""
Measure full analysis pipeline latency from AnalysisReport.performance_summary.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.orchestrator_agent import OrchestratorAgent
from models.data_models import AnalysisReport


async def measure_pipeline_stages(ticker: str, orchestrator: Optional[OrchestratorAgent] = None) -> Dict[str, Any]:
    """
    Run one full async analysis and return structured timing (per stage + total).

    If ``orchestrator`` is omitted, a temporary :class:`~agents.orchestrator_agent.OrchestratorAgent`
    is constructed and closed after the run.

    Returns
    -------
    dict
        ``ticker``, ISO timestamps, ``total_pipeline_seconds``, and a ``stages`` list; or an error stub
        if :attr:`~models.data_models.AnalysisReport.performance_summary` is missing.
    """
    orch = orchestrator or OrchestratorAgent()
    try:
        report: AnalysisReport = await orch.run_analysis_async(ticker)
    finally:
        if orchestrator is None:
            orch.close()

    perf = report.performance_summary
    if perf is None:
        return {
            "ticker": ticker,
            "error": "no_performance_summary",
            "total_pipeline_seconds": None,
            "stages": [],
        }

    stages_out = [
        {
            "name": s.name,
            "duration_seconds": round(s.duration_seconds, 6),
            "succeeded": s.succeeded,
            "error": s.error,
        }
        for s in perf.stages
    ]

    return {
        "ticker": ticker,
        "started_at": perf.started_at.isoformat(),
        "finished_at": perf.finished_at.isoformat(),
        "total_pipeline_seconds": round(perf.total_duration_seconds, 6),
        "stages": stages_out,
    }
