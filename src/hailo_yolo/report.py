"""Findings shared by the command-line checks (hefcheck, doctor)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    level: str  # "error": won't work, "warning": works but wrong or slow, "info"
    message: str


def print_findings(findings: list[Finding]) -> int:
    """Print one line per finding; return exit code 0 if all are info, else 1."""
    for finding in findings:
        print(f"{finding.level.upper():8} {finding.message}")
    return 1 if any(f.level != "info" for f in findings) else 0
