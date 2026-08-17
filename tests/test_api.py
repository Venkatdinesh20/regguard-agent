"""HTTP contract tests."""

from __future__ import annotations


class TestHealth:
    def test_health_reports_configuration(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["llm_provider"] == "stub"


class TestValidation:
    def test_malformed_case_id_is_rejected_before_any_work(self, client):
        response = client.post(
            "/investigations",
            json={"case_id": "bad case id", "query": "Investigate customer C001"},
        )
        assert response.status_code == 422

    def test_short_query_is_rejected(self, client):
        response = client.post(
            "/investigations", json={"case_id": "CASE-1", "query": "hi"}
        )
        assert response.status_code == 422

    def test_unknown_thread_returns_404(self, client):
        assert client.get("/investigations/does-not-exist").status_code == 404

    def test_approval_requires_an_approver(self, client):
        response = client.post(
            "/investigations/any-thread/approval", json={"approved": True}
        )
        assert response.status_code == 422


class TestInvestigationLifecycle:
    def test_high_risk_case_pauses_then_resumes(self, client):
        opened = client.post(
            "/investigations",
            json={
                "case_id": "CASE-API-1",
                "query": "Investigate unusual cash deposits for customer C001",
                "customer_id": "C001",
                "lookback_days": 60,
            },
        )
        assert opened.status_code == 202  # accepted, awaiting authorisation
        body = opened.json()
        assert body["status"] == "awaiting_approval"
        assert body["report"]["risk_level"] == "HIGH"
        assert body["pending_approval"]["requires_sar_filing"] is True
        thread_id = body["thread_id"]

        read = client.get(f"/investigations/{thread_id}")
        assert read.status_code == 200
        assert read.json()["status"] == "awaiting_approval"

        approved = client.post(
            f"/investigations/{thread_id}/approval",
            json={
                "approved": True,
                "approver": "compliance.officer@bank.example",
                "notes": "Structuring pattern confirmed; proceed to STR.",
            },
        )
        assert approved.status_code == 200
        resumed = approved.json()
        assert resumed["status"] == "approved"
        assert resumed["approval"]["approver"] == "compliance.officer@bank.example"

    def test_low_risk_case_completes_in_one_call(self, client):
        response = client.post(
            "/investigations",
            json={
                "case_id": "CASE-API-2",
                "query": "Review customer C002 following a monitoring alert",
                "customer_id": "C002",
                "lookback_days": 60,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "reported"
        assert body["report"]["risk_level"] == "LOW"
        assert body["pending_approval"] is None

    def test_response_exposes_the_full_audit_trail(self, client):
        response = client.post(
            "/investigations",
            json={
                "case_id": "CASE-API-3",
                "query": "Assess wire activity and source of funds for customer C003",
                "customer_id": "C003",
                "lookback_days": 60,
            },
        )
        body = response.json()
        assert [d["next_agent"] for d in body["decisions"]][-1] == "FINISH"
        assert body["route_history"] == ["CUSTOMER", "TRANSACTION", "FRAUD", "POLICY"]
        assert all(finding["summary"] for finding in body["findings"])
        assert body["steps_used"] == 5
