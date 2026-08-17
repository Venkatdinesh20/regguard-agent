"""Authentication, authorisation, and who an approval is attributed to."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.core.security import Principal, Role, parse_api_tokens, principal_for_token

ANALYST_TOKEN = "tok-analyst"
APPROVER_TOKEN = "tok-approver"
TOKENS = (
    f"{ANALYST_TOKEN}:a.analyst@bank.example:analyst,"
    f"{APPROVER_TOKEN}:c.officer@bank.example:approver"
)

HIGH_RISK_CASE = {
    "case_id": "CASE-AUTH",
    "query": "Investigate unusual cash deposits for customer C001",
    "customer_id": "C001",
    "lookback_days": 60,
}


@pytest.fixture
def secured_client(monkeypatch) -> Iterator[TestClient]:
    """A client with authentication switched on."""
    from app.core.llm import reset_chat_model_cache
    from app.graph.build import reset_graph_cache

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_TOKENS", TOKENS)
    get_settings.cache_clear()
    reset_chat_model_cache()
    reset_graph_cache()

    from app.api import app

    with TestClient(app) as client:
        yield client


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestTokenParsing:
    def test_role_defaults_to_analyst(self):
        tokens = parse_api_tokens("t1:someone@bank.example")
        assert tokens["t1"] == Principal(name="someone@bank.example", role=Role.ANALYST)

    def test_explicit_roles_are_parsed(self):
        tokens = parse_api_tokens(TOKENS)
        assert tokens[APPROVER_TOKEN].can_approve is True
        assert tokens[ANALYST_TOKEN].can_approve is False

    def test_blank_configuration_yields_no_principals(self):
        assert parse_api_tokens("") == {}
        assert parse_api_tokens("  ,  ") == {}

    @pytest.mark.parametrize(
        "raw",
        [
            "justatoken",
            "t1:name:analyst:extra",
            ":name:analyst",
            "t1::analyst",
            "t1:name:wizard",
            "t1:a:analyst,t1:b:approver",
        ],
    )
    def test_malformed_configuration_is_rejected_loudly(self, raw):
        """Silently granting or denying access is the worst outcome."""
        with pytest.raises(ConfigurationError):
            parse_api_tokens(raw)

    def test_unknown_token_resolves_to_nobody(self):
        settings = Settings.model_validate({"auth_enabled": True, "api_tokens": TOKENS})
        assert principal_for_token("nope", settings) is None
        assert principal_for_token("", settings) is None
        assert principal_for_token(APPROVER_TOKEN, settings) is not None


class TestConfigurationGuards:
    def test_auth_enabled_without_tokens_is_refused(self):
        with pytest.raises(ValidationError, match="API_TOKENS"):
            Settings.model_validate({"auth_enabled": True})

    def test_production_requires_authentication(self):
        with pytest.raises(ValidationError, match="AUTH_ENABLED=true"):
            Settings.model_validate(
                {
                    "environment": "production",
                    "llm_provider": "openai",
                    "openai_api_key": "sk-test",
                }
            )

    def test_production_refuses_the_stub_provider(self):
        with pytest.raises(ValidationError, match="cannot run with LLM_PROVIDER=stub"):
            Settings.model_validate(
                {
                    "environment": "production",
                    "auth_enabled": True,
                    "api_tokens": TOKENS,
                }
            )

    def test_a_valid_production_configuration_is_accepted(self):
        settings = Settings.model_validate(
            {
                "environment": "production",
                "auth_enabled": True,
                "api_tokens": TOKENS,
                "llm_provider": "openai",
                "openai_api_key": "sk-test",
            }
        )
        assert settings.is_production is True


class TestSecuredEndpoints:
    def test_health_stays_public_and_reports_the_posture(self, secured_client):
        body = secured_client.get("/health").json()
        assert body["auth_enabled"] is True
        assert body["time_anchor"] == "dataset"

    def test_no_token_is_rejected(self, secured_client):
        response = secured_client.post("/investigations", json=HIGH_RISK_CASE)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_bad_token_is_rejected(self, secured_client):
        response = secured_client.post(
            "/investigations", json=HIGH_RISK_CASE, headers=_auth("wrong")
        )
        assert response.status_code == 401

    def test_reading_a_case_requires_a_token(self, secured_client):
        assert secured_client.get("/investigations/anything").status_code == 401

    def test_analyst_can_open_a_case_but_not_authorise_it(self, secured_client):
        opened = secured_client.post(
            "/investigations", json=HIGH_RISK_CASE, headers=_auth(ANALYST_TOKEN)
        )
        assert opened.status_code == 202
        thread_id = opened.json()["thread_id"]

        denied = secured_client.post(
            f"/investigations/{thread_id}/approval",
            json={"approved": True},
            headers=_auth(ANALYST_TOKEN),
        )
        assert denied.status_code == 403

    def test_approver_authorises_and_the_token_supplies_the_identity(
        self, secured_client
    ):
        opened = secured_client.post(
            "/investigations", json=HIGH_RISK_CASE, headers=_auth(ANALYST_TOKEN)
        )
        thread_id = opened.json()["thread_id"]

        approved = secured_client.post(
            f"/investigations/{thread_id}/approval",
            json={"approved": True, "notes": "Structuring confirmed."},
            headers=_auth(APPROVER_TOKEN),
        )
        assert approved.status_code == 200
        assert approved.json()["approval"]["approver"] == "c.officer@bank.example"

    def test_a_body_approver_cannot_impersonate_someone_else(self, secured_client):
        """The audit trail records the authenticated principal, nothing else."""
        opened = secured_client.post(
            "/investigations", json=HIGH_RISK_CASE, headers=_auth(APPROVER_TOKEN)
        )
        thread_id = opened.json()["thread_id"]

        approved = secured_client.post(
            f"/investigations/{thread_id}/approval",
            json={"approved": True, "approver": "someone.else@bank.example"},
            headers=_auth(APPROVER_TOKEN),
        )
        assert approved.status_code == 200
        assert approved.json()["approval"]["approver"] == "c.officer@bank.example"


class TestUnauthenticatedMode:
    """The out-of-the-box path stays usable, but must name its reviewer."""

    def test_approver_is_required_when_auth_is_disabled(self, client):
        opened = client.post("/investigations", json=HIGH_RISK_CASE)
        thread_id = opened.json()["thread_id"]

        response = client.post(
            f"/investigations/{thread_id}/approval", json={"approved": True}
        )
        assert response.status_code == 422
        assert "approver is required" in response.json()["detail"]

    def test_body_approver_is_accepted_when_auth_is_disabled(self, client):
        opened = client.post("/investigations", json=HIGH_RISK_CASE)
        thread_id = opened.json()["thread_id"]

        response = client.post(
            f"/investigations/{thread_id}/approval",
            json={"approved": True, "approver": "local.dev"},
        )
        assert response.status_code == 200
        assert response.json()["approval"]["approver"] == "local.dev"
