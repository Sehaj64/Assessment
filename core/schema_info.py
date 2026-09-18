"""
Schema definition and domain metadata for the Support Tickets Dataset.
Used for injecting context into LLM prompts for reliable Text-to-SQL generation.
"""

SCHEMA_DDL = """
CREATE TABLE tickets (
    ticket_id TEXT PRIMARY KEY,          -- Unique identifier (e.g. 'TKT-001')
    created_at TEXT NOT NULL,           -- Timestamp in 'YYYY-MM-DD HH:MM' format
    category TEXT NOT NULL,             -- Issue category: 'Billing', 'Technical', 'General'
    priority TEXT NOT NULL,             -- Urgency level: 'Low', 'Medium', 'High', 'Critical'
    status TEXT NOT NULL,               -- Ticket status: 'Open', 'Resolved', 'Escalated'
    response_time_hrs REAL NOT NULL,    -- Hours from creation to first agent response
    resolution_time_hrs REAL,           -- Hours from creation to resolution (NULL if unresolved)
    agent_id TEXT NOT NULL,             -- Assigned agent identifier (e.g. 'AGT-01' to 'AGT-12')
    customer_rating INTEGER,            -- Customer rating 1-5 (NULL if unresolved)
    issue_summary TEXT NOT NULL,        -- Free-text brief summary of the issue
    is_resolved INTEGER NOT NULL,       -- 1 if status = 'Resolved', else 0
    is_unresolved INTEGER NOT NULL,     -- 1 if status in ('Open', 'Escalated'), else 0
    created_date TEXT NOT NULL,         -- Date in 'YYYY-MM-DD'
    created_year_month TEXT NOT NULL    -- Month in 'YYYY-MM'
);
"""

COLUMN_DESCRIPTIONS = {
    "ticket_id": "Unique ticket ID string (e.g., TKT-001 to TKT-500)",
    "created_at": "Creation timestamp in 'YYYY-MM-DD HH:MM' format",
    "category": "One of: 'Billing', 'Technical', 'General'",
    "priority": "One of: 'Low', 'Medium', 'High', 'Critical'",
    "status": "One of: 'Open', 'Resolved', 'Escalated'",
    "response_time_hrs": "Time to first response in hours (float >= 0)",
    "resolution_time_hrs": "Time to resolve in hours (float, NULL if status != 'Resolved')",
    "agent_id": "Identifier of assigned agent (AGT-01 to AGT-12)",
    "customer_rating": "Satisfaction score 1-5 (NULL if status != 'Resolved')",
    "issue_summary": "Description of the customer issue",
    "is_resolved": "1 if status = 'Resolved', 0 otherwise",
    "is_unresolved": "1 if status in ('Open', 'Escalated'), 0 otherwise",
    "created_date": "Date part of created_at ('YYYY-MM-DD')",
    "created_year_month": "Year and month part ('YYYY-MM')"
}

CATEGORY_VALUES = ["Billing", "Technical", "General"]
PRIORITY_VALUES = ["Low", "Medium", "High", "Critical"]
STATUS_VALUES = ["Open", "Resolved", "Escalated"]
AGENTS = [f"AGT-{i:02d}" for i in range(1, 13)]

DOMAIN_RULES = """
Important Domain & SQL Guidelines for SQLite:
1. 'unresolved' tickets are tickets where status IN ('Open', 'Escalated') or is_unresolved = 1.
2. 'resolved' tickets are tickets where status = 'Resolved' or is_resolved = 1.
3. 'resolution_time_hrs' and 'customer_rating' are NULL for unresolved tickets. Use 'WHERE resolution_time_hrs IS NOT NULL' or 'WHERE status = \"Resolved\"' when calculating averages or aggregations on these columns.
4. Ratings range from 1 (lowest) to 5 (highest).
5. For date queries:
   - Specific month (e.g. 'this month', 'March 2024'): use created_year_month = '2024-03' or strftime('%Y-%m', created_at) = '2024-03'. The dataset spans January 2024 to March 2024.
   - Latest month in dataset is '2024-03'.
6. Only return valid SQLite-compatible SELECT statements. Never execute DROP, UPDATE, INSERT, or DELETE.
7. Use round(..., 2) for averages where appropriate.
"""
