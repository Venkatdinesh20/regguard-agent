"""The CLI entrypoint and the audit log format."""

from __future__ import annotations

import json
import logging

from app.core.logging import (
    CaseIdFilter,
    JsonFormatter,
    case_context,
    configure_logging,
    current_case_id,
)
from app.main import main


class TestCli:
    def test_single_customer_run_prints_the_audit_trail(self, capsys):
        exit_code = main(
            ["--case-id", "CASE-CLI-1", "--customer", "C002", "--lookback-days", "60"]
        )
        output = capsys.readouterr().out

        assert exit_code == 0
        assert "CASE-CLI-1" in output
        assert "Control flow chosen by the LLM supervisor" in output
        assert "Risk level: LOW" in output

    def test_high_risk_run_stops_at_the_authorisation_gate_by_default(self, capsys):
        """Without --auto-approve the CLI must not simulate a human."""
        main(["--case-id", "CASE-CLI-2", "--customer", "C001", "--lookback-days", "60"])
        output = capsys.readouterr().out

        assert "PAUSED FOR HUMAN AUTHORISATION" in output
        assert "R01_STRUCTURING" in output
        assert "Human decision" not in output

    def test_auto_approve_simulates_the_reviewer_and_completes_the_case(self, capsys):
        main(
            [
                "--case-id",
                "CASE-CLI-2b",
                "--customer",
                "C001",
                "--lookback-days",
                "60",
                "--auto-approve",
            ]
        )
        output = capsys.readouterr().out

        assert "PAUSED FOR HUMAN AUTHORISATION" in output
        assert "Human decision: APPROVED" in output

    def test_free_text_query_is_accepted(self, capsys):
        exit_code = main(
            ["--case-id", "CASE-CLI-3", "--query", "Is customer C004 laundering funds?"]
        )
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "CASE-CLI-3" in output


class TestAuditLogging:
    def test_records_are_json_and_carry_the_case_id(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="supervisor.decision",
            args=(),
            exc_info=None,
        )
        record.case_id = "CASE-LOG"
        record.dispatched = "FRAUD"

        payload = json.loads(formatter.format(record))
        assert payload["message"] == "supervisor.decision"
        assert payload["case_id"] == "CASE-LOG"
        assert payload["dispatched"] == "FRAUD"
        assert "timestamp" in payload

    def test_case_context_binds_and_resets_the_correlation_id(self):
        assert current_case_id() == "-"

        with case_context("CASE-CTX"):
            assert current_case_id() == "CASE-CTX"
            record = logging.LogRecord(
                name="app.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="inside",
                args=(),
                exc_info=None,
            )
            assert CaseIdFilter().filter(record) is True
            assert record.case_id == "CASE-CTX"

        assert current_case_id() == "-"

    def test_text_format_is_available_for_local_development(self):
        configure_logging("DEBUG", "text")
        handler = logging.getLogger().handlers[0]
        assert not isinstance(handler.formatter, JsonFormatter)
        configure_logging("WARNING", "text")
