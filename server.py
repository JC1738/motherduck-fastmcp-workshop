"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import ask_choice, caller_email, read_choice

mcp = FastMCP("sf-311")
log = logging.getLogger("sf311.server")
# Nothing configures the root logger on FastMCP Gateway, so INFO lines would be dropped.
# Give this logger its own handler; it still propagates, so pytest's caplog sees it too.
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)

HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")
LAST_OPENED = max(r.opened for r in REQUESTS)


def mask_address(address: str) -> str:
    """Cut a street address to its block: "994 HARRISON ST, ..." -> "900 block of HARRISON ST"."""
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    row = request.as_dict()
    row["address"] = mask_address(request.address)
    return row

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
    return [_public(r) for r in rows[:limit]]


@mcp.tool(annotations=READ_ONLY)
def get_request(request_id: str) -> dict:
    """Return one 311 request by its request_id, with every field.

    Use this after search_requests to look at a single request in detail.
    """
    request = BY_ID.get(request_id.strip())
    log.info("get_request request_id=%s caller=%s", request_id, caller_email() or "local")
    if request is None:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find real request_ids.")
    return _public(request)


@mcp.tool(annotations=READ_ONLY)
def open_backlog(service: str | None = None, neighborhood: str | None = None) -> dict:
    """How many requests opened this week are still open, optionally for one service and official neighborhood.

    Returns open, closed, total, share_open (0 to 1), and
    oldest_open_hours: how long the oldest still-open request has waited,
    measured to the last request in the snapshot. Use this for "still open",
    "backlog", or "unresolved" questions in one call. Use count_requests
    with group_by="status" to compare backlogs across many groups, and
    median_close_hours for how fast requests close.
    """
    rows = _matching(service, neighborhood)
    still_open = [r for r in rows if r.status == "Open"]
    oldest = min((r.opened for r in still_open), default=None)
    return {
        "open": len(still_open),
        "closed": len(rows) - len(still_open),
        "total": len(rows),
        "share_open": round(len(still_open) / len(rows), 3) if rows else None,
        "oldest_open_hours": round((LAST_OPENED - oldest).total_seconds() / 3600, 1) if oldest else None,
    }


def _local_area(name: str) -> tuple[list[Request], list[str]]:
    """Requests in a local area, and the official neighborhoods it overlaps (largest first)."""
    if name not in LOCAL_NAMES:
        suggestions = closest(name, NEIGHBORHOODS + LOCAL_NAMES)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolError(f"Unknown neighborhood name {name!r}.{hint}")
    rows = [r for r in REQUESTS if r.local_name == name]
    counts = Counter(r.neighborhood for r in rows if r.neighborhood)
    parts = [n for n, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return rows, parts


@mcp.tool(annotations=READ_ONLY)
def count_in_neighborhood(name: str, ctx: Context, boundary: str | None = None) -> int | str:
    """Count requests in a neighborhood, by its official name or the local name people use.

    Official names (see dataset_info) are counted directly. A local name such
    as "Lower Nob Hill" can overlap several official neighborhoods; then the
    user is asked which part to count. Pass `boundary` to skip the question:
    one of the overlapped official neighborhoods, or "all". Returns a number,
    or a short message if the user declined. For the split across every part
    at once, use local_area_breakdown.
    """
    if name in NEIGHBORHOODS:
        return sum(1 for r in REQUESTS if r.neighborhood == name)
    rows, parts = _local_area(name)
    if len(parts) <= 1:
        return len(rows)

    all_label = f"All of {name}"
    if boundary is None:
        answer = read_choice(ctx, "boundary")
        if answer is None:
            return ask_choice("boundary", f"{name} overlaps {len(parts)} official neighborhoods. Which part should I count?", parts + [all_label])
        if not answer.accepted:
            return f"Nothing was counted: no part of {name} was chosen. Call again with boundary set to one of {parts} or 'all'."
        boundary = answer.value

    if boundary in ("all", all_label):
        return len(rows)
    if boundary not in parts:
        raise ToolError(f"{boundary!r} is not part of {name}. Use one of {parts}, or 'all'.")
    return sum(1 for r in rows if r.neighborhood == boundary)


@mcp.tool(annotations=READ_ONLY)
def local_area_breakdown(name: str) -> dict:
    """Split a local neighborhood name across the official neighborhoods it overlaps.

    Returns total and parts: each official neighborhood with its count and
    share (0 to 1), largest first. Use this for "which part holds the most"
    or "what share" questions about a local name; it asks nothing. Use
    count_in_neighborhood for a single count.
    """
    rows, parts = _local_area(name)
    counts = Counter(r.neighborhood for r in rows if r.neighborhood)
    return {
        "name": name,
        "total": len(rows),
        "parts": [{"neighborhood": n, "count": counts[n], "share": round(counts[n] / len(rows), 3)} for n in parts],
    }


if __name__ == "__main__":
    mcp.run()
