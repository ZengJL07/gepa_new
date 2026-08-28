"""Thin REST client for AgentGym env servers (raw ``requests``, no agentenv dep).

The servers are unauthenticated localhost dev services. A configured proxy
(``http_proxy``) must be bypassed for them — but passing ``proxies={}`` is NOT
enough: with the default ``trust_env=True``, requests still merges proxies from
the environment, and its ``no_proxy`` matcher does not understand glob forms like
``127.*``, so ``127.0.0.1`` gets sent to the proxy and fails (e.g. 502). We use a
session with ``trust_env=False`` to ignore ambient proxy/netrc entirely — the
equivalent of ``curl --noproxy '*'``.

Any ``{"error": ...}`` payload or non-2xx status raises :class:`EnvError`.
"""

import os
import threading
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from examples.tool_loop.envs.base import EnvError

# One process-wide session that ignores environment proxy settings. Lazily built
# and guarded by a lock (requests.Session is not guaranteed thread-safe to build,
# though reuse across threads for simple requests is fine).
_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()

# Connection pool size. requests' default is 10, which is smaller than our
# episode concurrency (15): with pool_maxsize=10 and pool_block=False, urllib3
# discards the surplus connections instead of queueing, so threads 11..15 open a
# fresh socket per request and the pool thrashes. Sized to the worker count.
_POOL_SIZE = int(os.environ.get("TOOL_LOOP_HTTP_POOL_SIZE", "32"))

# Retries for "the server closed an idle pooled connection" — the dominant
# failure here. uvicorn's timeout_keep_alive is 5s, but an AlfWorld episode can
# leave a connection idle far longer than that while waiting on the solver LLM
# (up to 20 turns of generation between two /step calls). The server then closes
# its end; the client only finds out when it writes to the dead socket, which
# surfaces as ConnectionResetError(104) / "Connection aborted".
#
# This is safe to retry despite POST not being idempotent in general: urllib3
# only retries when the request failed *before* a response was read, so the
# server either never received it or never answered. A retried /step therefore
# cannot double-apply an action.
_RETRIES = Retry(
    total=int(os.environ.get("TOOL_LOOP_HTTP_RETRIES", "4")),
    connect=4,
    read=2,
    backoff_factor=0.5,  # 0s, 0.5s, 1s, 2s
    status_forcelist=(502, 503, 504),
    allowed_methods=frozenset({"GET", "POST"}),
    raise_on_status=False,
)


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                s.trust_env = False  # ignore http_proxy/no_proxy/netrc for localhost dev servers
                adapter = HTTPAdapter(
                    pool_connections=_POOL_SIZE,
                    pool_maxsize=_POOL_SIZE,
                    max_retries=_RETRIES,
                    pool_block=False,
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _SESSION = s
    return _SESSION


def _check(resp: requests.Response, base: str, path: str) -> dict[str, Any]:
    if resp.status_code >= 400:
        raise EnvError(f"{base}{path} -> HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise EnvError(f"{base}{path} -> non-JSON response: {resp.text[:200]}") from e
    if isinstance(data, dict) and "error" in data:
        raise EnvError(f"{base}{path} -> server error: {data['error']}")
    return data


def post(base: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 300.0) -> dict[str, Any]:
    """POST JSON to ``base+path`` and return the parsed dict (raises EnvError on failure)."""
    resp = _session().post(f"{base}{path}", json=payload or {}, timeout=timeout)
    return _check(resp, base, path)


def get(base: str, path: str, params: dict[str, Any] | None = None, *, timeout: float = 300.0) -> dict[str, Any]:
    """GET ``base+path`` and return the parsed dict (raises EnvError on failure)."""
    resp = _session().get(f"{base}{path}", params=params or {}, timeout=timeout)
    return _check(resp, base, path)
