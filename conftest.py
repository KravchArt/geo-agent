"""Session-wide test setup.

Tests must not depend on the machine's network configuration. httpx reads
HTTP_PROXY/ALL_PROXY from the environment whenever a client is built without
``trust_env=False``, so a developer running behind a proxy gets different
results from CI — and an ``ALL_PROXY=socks://...`` makes httpx refuse to
construct a client at all, failing tests that never touch the network.

Clearing the variables here makes the suite hermetic: same behaviour on every
machine, no per-developer workarounds. Application code sets ``trust_env=False``
explicitly (see backend/app/main.py and backend/app/llm/vllm.py), so this only
covers clients that tests build themselves.
"""

import os

for _proxy_var in (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "FTP_PROXY",
    "ftp_proxy",
):
    os.environ.pop(_proxy_var, None)
