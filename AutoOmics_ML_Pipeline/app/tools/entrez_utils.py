"""
Centralized Entrez configuration, rate-limit throttle, and request wrappers.

DESIGN: throttling happens BEFORE each request is dispatched, not after the
response is received.  This prevents concurrent threads from launching multiple
requests simultaneously before any post-read sleep fires.

Usage in all Entrez-backed tool files — replace raw Entrez calls with wrappers:

    from app.tools.entrez_utils import entrez_esearch, entrez_efetch, entrez_esummary

    handle = entrez_esearch(db="pubmed", term="...", retmode="xml", retmax=5)
    record = Entrez.read(handle)
    handle.close()
    # No explicit sleep needed — throttle fires pre-dispatch

Rate limits (NCBI policy):
  Without NCBI_API_KEY : 3 req/s max → 1.5 s minimum gap (conservative default)
  With    NCBI_API_KEY : 10 req/s max → 0.5 s minimum gap (conservative default)

Both gaps are configurable via environment variables:
  ENTREZ_SLEEP_NO_KEY   (default "1.5")
  ENTREZ_SLEEP_WITH_KEY (default "0.5")

The throttle is process-wide and thread-safe.  Concurrent threads from a
ThreadPoolExecutor share the same counter — requests are serialized through
the lock so they never launch simultaneously.

Note: socket.setdefaulttimeout is intentionally NOT set here; it is a
process-wide side effect that can interfere with unrelated networking.
Entrez.timeout is sufficient for all NCBI HTTP calls in this project.
"""
import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()

from Bio import Entrez

# ------------------------------------------------------------------
# Configuration — read once at import time
# ------------------------------------------------------------------
ENTREZ_EMAIL   = os.getenv("ENTREZ_EMAIL", "")
ENTREZ_API_KEY = os.getenv("NCBI_API_KEY", "")      # blank string = no key
ENTREZ_TIMEOUT = int(os.getenv("ENTREZ_TIMEOUT", "45"))

_SLEEP_NO_KEY   = float(os.getenv("ENTREZ_SLEEP_NO_KEY",   "1.5"))
_SLEEP_WITH_KEY = float(os.getenv("ENTREZ_SLEEP_WITH_KEY", "0.5"))

# ------------------------------------------------------------------
# Apply configuration to Biopython Entrez (runs once on import)
# ------------------------------------------------------------------
Entrez.email   = ENTREZ_EMAIL
Entrez.timeout = ENTREZ_TIMEOUT
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY

# ------------------------------------------------------------------
# Process-wide throttle state
# ------------------------------------------------------------------
_throttle_lock:  threading.Lock = threading.Lock()
_last_call_time: float          = 0.0


def _pre_dispatch() -> None:
    """Acquire the shared throttle and wait before dispatching an Entrez request.

    Acquires the lock, waits if needed to honour the minimum inter-request gap,
    records the dispatch time, then releases the lock.

    Gap defaults:
      1.5 s — no NCBI_API_KEY  (safe below NCBI's 3 req/s limit)
      0.5 s — NCBI_API_KEY set (safe below NCBI's 10 req/s limit)
    Override via ENTREZ_SLEEP_NO_KEY / ENTREZ_SLEEP_WITH_KEY env vars.
    """
    global _last_call_time
    min_gap = _SLEEP_WITH_KEY if ENTREZ_API_KEY else _SLEEP_NO_KEY
    with _throttle_lock:
        elapsed = time.perf_counter() - _last_call_time
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        _last_call_time = time.perf_counter()


# ------------------------------------------------------------------
# Throttled Entrez wrappers
# All Entrez-backed tool files must use these instead of Entrez.* directly.
# ------------------------------------------------------------------

def entrez_esearch(**kwargs):
    """Throttled Entrez.esearch — use in place of Entrez.esearch(...)."""
    _pre_dispatch()
    return Entrez.esearch(**kwargs)


def entrez_esummary(**kwargs):
    """Throttled Entrez.esummary — use in place of Entrez.esummary(...)."""
    _pre_dispatch()
    return Entrez.esummary(**kwargs)


def entrez_efetch(**kwargs):
    """Throttled Entrez.efetch — use in place of Entrez.efetch(...)."""
    _pre_dispatch()
    return Entrez.efetch(**kwargs)
