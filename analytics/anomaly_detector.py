"""
Multi-factor Anomaly Detection Engine for customer support tickets.
Combines operational SLA rule checks, statistical IQR/Z-Score bounds,
and Scikit-Learn Isolation Forest machine learning.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.database import DatabaseManager


@dataclass
class AnomalyRecord:
    ticket_id: str
    category: str
    priority: str
    status: str
    agent_id: str
    created_at: str
    anomaly_type: str        # 'SLA_BREACH', 'STATISTICAL_OUTLIER', 'ML_ANOMALY', 'QUALITY_DEFECT'
    severity: str            # 'CRITICAL', 'HIGH', 'MEDIUM'
    reason: str              # Explanatory text
    metric_name: str
    metric_value: float
    threshold: float
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnomalyDetector:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()

    def get_dataframe(self) -> pd.DataFrame:
        conn = self.db.get_connection()
        return pd.read_sql_query("SELECT * FROM tickets", conn)

    def detect_all(self, severity_filter: Optional[str] = None) -> Dict[str, Any]:
        """Runs all detection pipelines and returns combined, deduplicated anomalies."""
        df = self.get_dataframe()
        anomalies: List[AnomalyRecord] = []

        # 1. Operational SLA Rules
        sla_anomalies = self._detect_sla_breaches(df)
        anomalies.extend(sla_anomalies)

        # 2. Statistical Outliers (Resolution & Response Times)
        stat_anomalies = self._detect_statistical_outliers(df)
        anomalies.extend(stat_anomalies)

        # 3. Machine Learning Isolation Forest (Multivariate)
        ml_anomalies = self._detect_ml_anomalies(df)
        anomalies.extend(ml_anomalies)

        # 4. Quality & Satisfaction Discrepancies
        quality_anomalies = self._detect_quality_anomalies(df)
        anomalies.extend(quality_anomalies)

        # Deduplicate by ticket_id and highest severity
        deduped: Dict[str, AnomalyRecord] = {}
        severity_order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}

        for item in anomalies:
            tid = item.ticket_id
            if tid not in deduped:
                deduped[tid] = item
            else:
                existing_score = severity_order.get(deduped[tid].severity, 0)
                new_score = severity_order.get(item.severity, 0)
                if new_score > existing_score:
                    deduped[tid] = item

        results = list(deduped.values())

        if severity_filter:
            results = [a for a in results if a.severity.upper() == severity_filter.upper()]

        # Sort by severity descending
        results.sort(key=lambda a: severity_order.get(a.severity, 0), reverse=True)

        summary = {
            "total_anomalies_flagged": len(results),
            "severity_breakdown": {
                "CRITICAL": sum(1 for a in results if a.severity == "CRITICAL"),
                "HIGH": sum(1 for a in results if a.severity == "HIGH"),
                "MEDIUM": sum(1 for a in results if a.severity == "MEDIUM"),
            },
            "type_breakdown": {
                "SLA_BREACH": sum(1 for a in results if a.anomaly_type == "SLA_BREACH"),
                "STATISTICAL_OUTLIER": sum(1 for a in results if a.anomaly_type == "STATISTICAL_OUTLIER"),
                "ML_ANOMALY": sum(1 for a in results if a.anomaly_type == "ML_ANOMALY"),
                "QUALITY_DEFECT": sum(1 for a in results if a.anomaly_type == "QUALITY_DEFECT"),
            },
            "anomalies": [a.to_dict() for a in results]
        }

        return summary

    def _detect_sla_breaches(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        """Detects tickets violating service-level targets."""
        records = []
        max_timestamp = pd.to_datetime(df["created_at"].max())

        for _, row in df.iterrows():
            created_dt = pd.to_datetime(row["created_at"])
            age_hours = (max_timestamp - created_dt).total_seconds() / 3600.0

            # Rule A: Unresolved High or Critical tickets open > 24 hours relative to dataset timeline
            if row["status"] in ["Open", "Escalated"] and row["priority"] in ["Critical", "High"]:
                if age_hours > 24.0:
                    records.append(AnomalyRecord(
                        ticket_id=row["ticket_id"],
                        category=row["category"],
                        priority=row["priority"],
                        status=row["status"],
                        agent_id=row["agent_id"],
                        created_at=row["created_at"],
                        anomaly_type="SLA_BREACH",
                        severity="CRITICAL" if row["priority"] == "Critical" else "HIGH",
                        reason=f"Unresolved {row['priority']} ticket pending for {age_hours:.1f} hours without resolution.",
                        metric_name="pending_age_hrs",
                        metric_value=round(age_hours, 1),
                        threshold=24.0,
                        recommended_action="Immediate escalation to tier-2 lead; trigger real-time notification to assigned agent."
                    ))

            # Rule B: Critical tickets taking > 12 hours to resolve
            if row["status"] == "Resolved" and row["priority"] == "Critical" and pd.notnull(row["resolution_time_hrs"]):
                if row["resolution_time_hrs"] > 12.0:
                    records.append(AnomalyRecord(
                        ticket_id=row["ticket_id"],
                        category=row["category"],
                        priority=row["priority"],
                        status=row["status"],
                        agent_id=row["agent_id"],
                        created_at=row["created_at"],
                        anomaly_type="SLA_BREACH",
                        severity="HIGH",
                        reason=f"Critical ticket resolution took {row['resolution_time_hrs']} hrs (Target SLA < 12 hrs).",
                        metric_name="resolution_time_hrs",
                        metric_value=float(row["resolution_time_hrs"]),
                        threshold=12.0,
                        recommended_action="Review post-mortem with agent; investigate root cause for delayed escalation."
                    ))

            # Rule C: Abnormally delayed first response time (> 4.5 hours)
            if row["response_time_hrs"] > 4.5:
                records.append(AnomalyRecord(
                    ticket_id=row["ticket_id"],
                    category=row["category"],
                    priority=row["priority"],
                    status=row["status"],
                    agent_id=row["agent_id"],
                    created_at=row["created_at"],
                    anomaly_type="SLA_BREACH",
                    severity="MEDIUM",
                    reason=f"First response delayed to {row['response_time_hrs']} hrs (Standard SLA < 4.0 hrs).",
                    metric_name="response_time_hrs",
                    metric_value=float(row["response_time_hrs"]),
                    threshold=4.0,
                    recommended_action="Assess agent queue backlog and balance incoming routing."
                ))

        return records

    def _detect_statistical_outliers(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        """Uses Interquartile Range (IQR) and Z-Scores to detect mathematical outliers."""
        records = []
        resolved_df = df[df["status"] == "Resolved"].copy()
        if resolved_df.empty:
            return records

        # Calculate IQR on resolution_time_hrs
        q1 = resolved_df["resolution_time_hrs"].quantile(0.25)
        q3 = resolved_df["resolution_time_hrs"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 2.0 * iqr  # Strong outlier boundary (~56.8 hrs)

        mean_res = resolved_df["resolution_time_hrs"].mean()
        std_res = resolved_df["resolution_time_hrs"].std()

        for _, row in resolved_df.iterrows():
            res_val = row["resolution_time_hrs"]
            z_score = (res_val - mean_res) / std_res if std_res > 0 else 0

            if res_val > upper_bound or z_score > 2.5:
                records.append(AnomalyRecord(
                    ticket_id=row["ticket_id"],
                    category=row["category"],
                    priority=row["priority"],
                    status=row["status"],
                    agent_id=row["agent_id"],
                    created_at=row["created_at"],
                    anomaly_type="STATISTICAL_OUTLIER",
                    severity="HIGH" if z_score > 3.0 else "MEDIUM",
                    reason=f"Abnormally long resolution time of {res_val:.1f} hrs (IQR Upper Bound: {upper_bound:.1f} hrs, Z={z_score:.2f}).",
                    metric_name="resolution_time_hrs",
                    metric_value=float(res_val),
                    threshold=round(upper_bound, 1),
                    recommended_action="Audit ticket handling history for blocked dependencies or agent training needs."
                ))

        return records

    def _detect_ml_anomalies(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        """Isolation Forest on (response_time, resolution_time, customer_rating) for multidimensional anomalies."""
        records = []
        resolved_df = df[(df["status"] == "Resolved") & df["resolution_time_hrs"].notnull() & df["customer_rating"].notnull()].copy()
        if len(resolved_df) < 20:
            return records

        features = resolved_df[["response_time_hrs", "resolution_time_hrs", "customer_rating"]].values
        # Fit Isolation Forest
        iso = IsolationForest(contamination=0.04, random_state=42)
        predictions = iso.fit_predict(features)
        anomaly_scores = iso.decision_function(features)

        resolved_df["is_anomaly"] = predictions
        resolved_df["anomaly_score"] = anomaly_scores

        anomalous_rows = resolved_df[resolved_df["is_anomaly"] == -1]

        for _, row in anomalous_rows.iterrows():
            records.append(AnomalyRecord(
                ticket_id=row["ticket_id"],
                category=row["category"],
                priority=row["priority"],
                status=row["status"],
                agent_id=row["agent_id"],
                created_at=row["created_at"],
                anomaly_type="ML_ANOMALY",
                severity="MEDIUM",
                reason=f"Multi-dimensional anomaly detected by Isolation Forest (Score: {row['anomaly_score']:.3f}, Response: {row['response_time_hrs']}h, Resol: {row['resolution_time_hrs']}h, Rating: {row['customer_rating']}).",
                metric_name="ml_anomaly_score",
                metric_value=round(float(row["anomaly_score"]), 3),
                threshold=0.0,
                recommended_action="Investigate complex edge-case interactions between response lag and customer satisfaction."
            ))

        return records

    def _detect_quality_anomalies(self, df: pd.DataFrame) -> List[AnomalyRecord]:
        """Flags premature closures or low ratings despite speedy resolution."""
        records = []
        resolved_df = df[df["status"] == "Resolved"]

        for _, row in resolved_df.iterrows():
            # Rating = 1 with resolution under 3 hours suggests rushed, unhelpful closure
            if row["customer_rating"] == 1 and row["resolution_time_hrs"] <= 3.0:
                records.append(AnomalyRecord(
                    ticket_id=row["ticket_id"],
                    category=row["category"],
                    priority=row["priority"],
                    status=row["status"],
                    agent_id=row["agent_id"],
                    created_at=row["created_at"],
                    anomaly_type="QUALITY_DEFECT",
                    severity="HIGH",
                    reason=f"Customer gave minimum rating (1/5) despite fast resolution ({row['resolution_time_hrs']} hrs). Potential improper ticket closure.",
                    metric_name="customer_rating",
                    metric_value=1.0,
                    threshold=2.0,
                    recommended_action="Perform QA audit on agent communication and reach out to customer for retention."
                ))

        return records
