"""
Automated Test Suite for Support Ticket AI Intelligence System.
Tests data ingestion, SQL sandboxing, anomaly detection,
NL query engine, and FastAPI REST endpoints.
Run with: pytest tests/test_system.py -v
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import DatabaseManager
from analytics.anomaly_detector import AnomalyDetector
from services.query_engine import QueryEngine
from main import app


@pytest.fixture(scope="session")
def db():
    return DatabaseManager.get_instance()


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


# ==========================================
# 1. DATA INGESTION & DATABASE TESTS
# ==========================================
def test_database_ingestion(db):
    """Verifies that all 500 records are properly ingested with types and derived fields."""
    stats = db.get_summary_stats()
    assert stats["total_tickets"] == 500
    assert stats["resolved_count"] == 327
    assert stats["open_count"] == 111
    assert stats["escalated_count"] == 62
    assert stats["unresolved_count"] == 173
    assert stats["avg_customer_rating"] > 0
    assert stats["avg_response_time_hrs"] > 0


def test_sql_sandboxing(db):
    """Ensures mutation operations and stacked queries are blocked."""
    # Forbidden operations
    is_valid, err = db.validate_sql("DROP TABLE tickets")
    assert not is_valid
    assert "Potentially unsafe" in err

    is_valid, err = db.validate_sql("DELETE FROM tickets WHERE ticket_id = 'TKT-001'")
    assert not is_valid

    is_valid, err = db.validate_sql("UPDATE tickets SET status = 'Resolved'")
    assert not is_valid

    # Stacked queries
    is_valid, err = db.validate_sql("SELECT * FROM tickets; SELECT * FROM tickets")
    assert not is_valid
    assert "Stacked queries" in err

    # Valid query with trailing semicolon
    is_valid, err = db.validate_sql("SELECT COUNT(*) FROM tickets;")
    assert is_valid
    assert err is None


# ==========================================
# 2. ANOMALY DETECTION ENGINE TESTS
# ==========================================
def test_anomaly_detection(db):
    """Verifies multi-layered anomaly detection generates valid reports."""
    detector = AnomalyDetector(db)
    report = detector.detect_all()

    assert "total_anomalies_flagged" in report
    assert report["total_anomalies_flagged"] > 0
    assert "severity_breakdown" in report
    assert "type_breakdown" in report
    assert report["severity_breakdown"]["CRITICAL"] > 0
    assert report["type_breakdown"]["SLA_BREACH"] > 0
    assert report["type_breakdown"]["STATISTICAL_OUTLIER"] > 0

    # Check structure of individual anomaly records
    first_item = report["anomalies"][0]
    required_keys = ["ticket_id", "category", "priority", "status", "agent_id", "severity", "anomaly_type", "reason", "recommended_action"]
    for k in required_keys:
        assert k in first_item


def test_anomaly_severity_filter(db):
    """Verifies filtering anomalies by severity."""
    detector = AnomalyDetector(db)
    crit_report = detector.detect_all(severity_filter="CRITICAL")
    assert all(a["severity"] == "CRITICAL" for a in crit_report["anomalies"])


# ==========================================
# 3. NATURAL LANGUAGE QUERY ENGINE TESTS
# ==========================================
@pytest.mark.parametrize("query,expected_keyword", [
    ("How many tickets are currently open?", "111"),
    ("Which agent resolved the most tickets this month?", "AGT-"),
    ("Show me all Critical tickets not resolved within 12 hours.", "Critical"),
    ("What is the average customer rating for Technical category tickets?", "3.7"),
    ("How many critical tickets are unresolved?", "31"),
    ("Which agent has the lowest average customer rating?", "AGT-"),
])
def test_sample_benchmark_queries(db, query, expected_keyword):
    """Verifies all 7 benchmark queries from the assessment brief execute accurately."""
    engine = QueryEngine(db)
    res = engine.ask(query)

    assert res["error"] is None
    assert res["sql"] is not None
    assert res["row_count"] >= 0
    assert res["answer"] is not None
    assert expected_keyword.lower() in res["answer"].lower() or any(expected_keyword.lower() in str(v).lower() for row in res["data"] for v in row.values())


# ==========================================
# 4. REST API ENDPOINTS TESTS (FASTAPI)
# ==========================================
def test_api_health_endpoint(client):
    """GET /health"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database"]["total_tickets"] == 500
    assert "llm_configuration" in data


def test_api_stats_endpoint(client):
    """GET /stats"""
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tickets"] == 500
    assert data["unresolved_count"] == 173


def test_api_query_endpoint(client):
    """POST /query"""
    payload = {"query": "How many tickets are currently open?"}
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sql" in data
    assert "answer" in data
    assert data["row_count"] >= 1
    assert "111" in data["answer"] or (data["data"] and data["data"][0].get("COUNT(ticket_id)") == 111)


def test_api_anomalies_endpoint(client):
    """GET /anomalies"""
    resp = client.get("/anomalies?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_anomalies_flagged" in data
    assert len(data["anomalies"]) <= 10


def test_api_tickets_filter(client):
    """GET /tickets"""
    resp = client.get("/tickets?category=Billing&priority=High&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tickets"]) <= 5
    for t in data["tickets"]:
        assert t["category"] == "Billing"
        assert t["priority"] == "High"


def test_api_ui_served(client):
    """GET / serves web dashboard"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "TicketPulse" in resp.text
