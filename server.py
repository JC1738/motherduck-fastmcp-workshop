"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest

mcp = FastMCP("sf-311")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10
READ_ONLY = {"readOnlyHint": True}

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}


def _check_name(value: str | None, choices: list[str], kind: str) -> None:
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown {kind} {value!r}.{hint} Call dataset_info for the exact names.")


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check_name(service, SERVICES, "service")
    _check_name(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations=READ_ONLY)
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations=READ_ONLY)
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: Literal["neighborhood", "service", "agency", "source", "weekday", "status"] | None = None,
) -> int | dict[str, int]:
    """Count 311 requests, optionally filtered by service and official neighborhood.

    Returns a single number, or with `group_by` a dict of group -> count,
    largest first, at most 50 groups. Use this for any "how many" or
    "which has the most" question; it is one call instead of paging through
    search_requests. Names must match dataset_info exactly.
    For median time to close, use median_close_hours.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations=READ_ONLY)
def median_close_hours(
    service: str | None = None,
    group_by: Literal["neighborhood", "agency"] | None = None,
) -> float | dict[str, float] | None:
    """Median hours from opened to closed, over closed requests only, rounded to 0.1.

    Returns one number, or with `group_by` a dict of group -> median hours,
    fastest first, leaving out groups with fewer than 10 closed requests.
    Returns null when nothing matching has closed. Use this for "how fast"
    or "which is slowest" questions; use count_requests for volumes.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {g: round(median(v), 1) for g, v in hours.items() if len(v) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations=READ_ONLY)
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: int = 20,
) -> list[dict]:
    """List individual 311 requests, newest first, filtered by service, official neighborhood, and status.

    Returns up to `limit` requests (1 to 50) as dicts with request_id,
    opened, closed, status, service, subtype, agency, neighborhood,
    local_name, address, and source. Use this to see examples. Do not use
    it to count or rank: it is capped at 50 rows, so use count_requests
    or median_close_hours instead. Use get_request for one known request_id.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}. For totals, use count_requests.")
    rows = sorted(_matching(service, neighborhood, status), key=lambda r: r.opened, reverse=True)
    return [r.as_dict() for r in rows[:limit]]


@mcp.tool(annotations=READ_ONLY)
def get_request(request_id: str) -> dict:
    """Return one 311 request by its request_id, with every field.

    Use this after search_requests to look at a single request in detail.
    """
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find real request_ids.")
    return request.as_dict()


if __name__ == "__main__":
    mcp.run()
