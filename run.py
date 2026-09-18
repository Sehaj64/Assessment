"""
Single-command launcher for Support Ticket AI Intelligence System.
Initializes the database, checks environment, and launches the FastAPI server and Web UI.
"""

import os
import sys
import uvicorn

def main():
    print("=" * 65)
    print("  TicketPulse AI - Support Ticket Intelligence System")
    print("  DOTMappers IT Pvt. Ltd. | AI Engineer Assessment Sprint")
    print("=" * 65)

    # Ingest database
    print("\n[1/3] Initializing SQLite Database from support_tickets.csv...")
    try:
        from core.database import DatabaseManager
        db = DatabaseManager.get_instance()
        stats = db.get_summary_stats()
        print(f"  [OK] Ingested {stats['total_tickets']} tickets into database.")
        print(f"       - Open / Escalated: {stats['unresolved_count']}")
        print(f"       - Resolved: {stats['resolved_count']}")
        print(f"       - Avg Customer Rating: {stats['avg_customer_rating']} / 5")
    except Exception as e:
        print(f"  [ERROR] Database initialization failed: {e}")
        sys.exit(1)

    # Detect LLM Provider
    print("\n[2/3] Checking Zero-Cost LLM Configuration...")
    from services.llm_client import LLMClient
    llm = LLMClient()
    info = llm.get_provider_info()
    print(f"  [OK] Active Provider: {info['provider'].upper()} (Model: {info['model']})")
    print(f"       - Zero-Cost Architecture: {info['is_zero_cost']}")

    # Start Server
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("\n[3/3] Launching FastAPI REST API & Interactive Web Dashboard...")
    print(f"  --> Web Dashboard:     http://localhost:{port}/")
    print(f"  --> Interactive Docs:  http://localhost:{port}/docs")
    print(f"  --> Alternative UI:    streamlit run app_streamlit.py")
    print("=" * 65 + "\n")

    uvicorn.run("main:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
