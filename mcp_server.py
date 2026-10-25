"""Standalone Model Context Protocol (MCP) server for the Incident Digest API.

Exposes operational runbook retrieval over standard input/output (stdio) so it
can be invoked as a subprocess by an MCP client.
"""

from __future__ import annotations

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - fallback for mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]

mcp = MCPServer("incident-digest-runbook-server")

RUNBOOKS: dict[str, str] = {
    "payments": (
        "Runbook: Restart the payment-gateway pod and check Stripe API status."
    ),
    "payment-service": (
        "Runbook: Restart the payment-gateway pod and check Stripe API status."
    ),
    "database": (
        "Runbook: Inspect active locks in pg_stat_activity, terminate long-running "
        "queries, and verify connection pooler (PgBouncer) health."
    ),
    "auth": (
        "Runbook: Check Redis session store connectivity and verify OAuth signing "
        "key rotation status."
    ),
    "redis": (
        "Runbook: Check Redis cluster node health on port 6379, inspect memory "
        "eviction policy, and trigger failover if primary is unreachable."
    ),
    "kafka": (
        "Runbook: Inspect consumer group lag, check broker disk utilization, and "
        "scale consumer worker replicas if lag exceeds threshold."
    ),
}


@mcp.tool()
def get_service_runbook(service_name: str) -> str:
    """Retrieve the operational runbook and remediation steps for a given service.

    Args:
        service_name: The name of the service experiencing an incident (e.g., 'payments',
            'database', 'auth', 'redis', 'kafka').

    Returns:
        A string containing the official runbook instructions for the specified service.
    """
    normalized = service_name.strip().lower()

    for key, runbook in RUNBOOKS.items():
        if key in normalized:
            return runbook

    return (
        f"Runbook: No specific runbook found for '{service_name}'. "
        "Check service health dashboard, inspect recent deployments, and escalate to "
        "the primary on-call engineer."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
