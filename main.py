"""
FastAPI Application serving REST endpoints for Natural Language Queries,
Anomaly Detection, Operational Statistics, and an interactive Web Dashboard.
"""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analytics.anomaly_detector import AnomalyDetector
from core.database import DatabaseManager
from services.llm_client import LLMClient
from services.query_engine import QueryEngine

app = FastAPI(
    title="Support Ticket AI Intelligence System",
    description="End-to-end AI system for NL querying, Text-to-SQL synthesis, and multi-factor anomaly detection over support tickets.",
    version="1.0.0"
)

# CORS middleware for local frontend development or API testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
db_manager = DatabaseManager.get_instance()
anomaly_detector = AnomalyDetector(db_manager)
query_engine = QueryEngine(db_manager)
llm_client = query_engine.llm

# Mount static files directory for the Web UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question to ask about tickets data", json_schema_extra={"example": "How many critical tickets are unresolved?"})


class QueryResponse(BaseModel):
    query: str
    sql: str
    data: List[Dict[str, Any]]
    row_count: int
    error: Optional[str] = None
    answer: str
    execution_time_ms: float
    llm_provider: str


@app.get("/health", summary="Health Check and System Diagnostics", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Returns server operational health, database row count, and active LLM configuration."""
    stats = db_manager.get_summary_stats()
    provider_info = llm_client.get_provider_info()
    return {
        "status": "healthy",
        "system": "Support Ticket AI Intelligence Engine",
        "version": "1.0.0",
        "database": {
            "status": "connected",
            "total_tickets": stats["total_tickets"],
            "date_range": stats["date_range"]
        },
        "llm_configuration": provider_info
    }


@app.post("/query", response_model=QueryResponse, summary="Natural Language Data Query", tags=["AI Query Engine"])
def query_tickets(request: QueryRequest) -> QueryResponse:
    """Answers natural language questions about tickets via Text-to-SQL and LLM synthesis."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = query_engine.ask(request.query)
    return QueryResponse(**result)


@app.get("/anomalies", summary="Detect and Flag Ticket Anomalies", tags=["Analytics & Anomaly Detection"])
def get_anomalies(
    severity: Optional[str] = Query(None, description="Filter by severity: 'CRITICAL', 'HIGH', 'MEDIUM'"),
    type: Optional[str] = Query(None, description="Filter by anomaly type: 'SLA_BREACH', 'STATISTICAL_OUTLIER', 'ML_ANOMALY', 'QUALITY_DEFECT'"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of anomaly records to return")
) -> Dict[str, Any]:
    """
    Detects operational SLA violations, statistical IQR/Z-Score outliers,
    and Scikit-Learn Isolation Forest anomalies.
    """
    report = anomaly_detector.detect_all(severity_filter=severity)
    
    anomalies = report["anomalies"]
    if type:
        anomalies = [a for a in anomalies if a["anomaly_type"].upper() == type.upper()]
    
    report["anomalies"] = anomalies[:limit]
    report["returned_count"] = len(report["anomalies"])
    return report


@app.get("/stats", summary="High-Level KPI Summary Statistics", tags=["Analytics & Anomaly Detection"])
def get_stats() -> Dict[str, Any]:
    """Returns aggregated KPIs: ticket counts, response/resolution averages, and category/priority distributions."""
    return db_manager.get_summary_stats()


@app.get("/tickets", summary="Filter and Search Tickets", tags=["Data"])
def get_tickets(
    category: Optional[str] = Query(None, description="Category filter (Billing, Technical, General)"),
    status: Optional[str] = Query(None, description="Status filter (Open, Resolved, Escalated)"),
    priority: Optional[str] = Query(None, description="Priority filter (Low, Medium, High, Critical)"),
    agent_id: Optional[str] = Query(None, description="Agent ID (AGT-01 to AGT-12)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """Retrieves tickets with filtering, pagination, and sorting."""
    clauses = []
    params = []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if priority:
        clauses.append("priority = ?")
        params.append(priority)
    if agent_id:
        clauses.append("agent_id = ?")
        params.append(agent_id)

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM tickets {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows, err = db_manager.execute_query(sql, tuple(params))
    if err:
        raise HTTPException(status_code=500, detail=err)

    # Count total matching
    count_sql = f"SELECT COUNT(*) FROM tickets {where_sql}"
    count_rows, _ = db_manager.execute_query(count_sql, tuple(params[:-2]))
    total_matching = count_rows[0]["COUNT(*)"] if count_rows else 0

    return {
        "total": total_matching,
        "limit": limit,
        "offset": offset,
        "tickets": rows
    }


@app.get("/", summary="Web Dashboard UI", include_in_schema=False)
def serve_dashboard():
    """Serves the interactive web interface."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to Support Ticket AI Intelligence System. Explore API docs at /docs."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on http://localhost:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
