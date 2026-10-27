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

ONCALL_ENGINEERS: dict[str, str] = {
    "payments": "On-Call Engineer: Alice Chen (+1-555-0101)",
    "payment-service": "On-Call Engineer: Alice Chen (+1-555-0101)",
    "database": "On-Call Engineer: Bob Martinez (+1-555-0102)",
    "auth": "On-Call Engineer: Priya Patel (+1-555-0103)",
    "redis": "On-Call Engineer: Marcus Johnson (+1-555-0104)",
    "kafka": "On-Call Engineer: Elena Rostova (+1-555-0105)",
}

SERVICE_HEALTH: dict[str, str] = {
    "payments": "Degraded: High latency and elevated 5xx error rate on checkout endpoints",
    "payment-service": "Degraded: High latency and elevated 5xx error rate on checkout endpoints",
    "database": "Unhealthy: Connection pool exhausted (98% active connections)",
    "auth": "Degraded: Intermittent token validation timeouts",
    "redis": "Healthy",
    "kafka": "Degraded: Consumer lag exceeding threshold on order-events topic",
}


@mcp.tool()
def get_service_runbook(service_name: str) -> str:
    """Retrieve the operational runbook and remediation steps for a given service.

    Use this tool when an incident or service degradation has been identified and
    you need official step-by-step instructions to diagnose, mitigate, or recover
    the affected service.

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


@mcp.tool()
def get_oncall_engineer(service_name: str) -> str:
    """Look up the primary on-call engineer's name and phone number for a given service.

    Use this tool when an incident requires human escalation, when you need to know
    who to page or contact for a specific service, or when assembling an incident
    response summary that includes point-of-contact details. Do NOT use this tool
    to check technical metrics or remediation steps.

    Args:
        service_name: The name of the service to look up the on-call contact for
            (e.g., 'payments', 'database', 'auth', 'redis', 'kafka').

    Returns:
        A string containing the full name and phone number of the on-call engineer
        responsible for the specified service.
    """
    normalized = service_name.strip().lower()

    for key, engineer in ONCALL_ENGINEERS.items():
        if key in normalized:
            return engineer

    return (
        f"On-Call Engineer: No dedicated on-call found for '{service_name}'. "
        "Fallback to Site Reliability Engineering Primary: Jordan Taylor (+1-555-0199)"
    )


@mcp.tool()
def check_service_health(service_name: str) -> str:
    """Check the current real-time operational health status of a given service.

    Use this tool FIRST when triaging a reported issue, verifying whether a service
    is actually experiencing degradation or downtime, or checking if dependent
    services are healthy vs. degraded before deciding on remediation or escalation.

    Args:
        service_name: The name of the service whose live health status should be checked
            (e.g., 'payments', 'database', 'auth', 'redis', 'kafka').

    Returns:
        A string describing the current health state of the service (e.g., 'Healthy',
        'Degraded: High latency...', or 'Unhealthy: ...').
    """
    normalized = service_name.strip().lower()

    for key, status in SERVICE_HEALTH.items():
        if key in normalized:
            return status

    return f"Healthy: No active alerts or anomalies detected for '{service_name}'."


if __name__ == "__main__":
    mcp.run(transport="stdio")
