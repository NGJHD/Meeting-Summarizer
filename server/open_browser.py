"""Wait for the server to answer, then open the browser. Launched by run.bat.

run.bat used to open the browser first and start uvicorn second, so the page
always loaded against a socket that was not listening yet and depended on the
frontend's retry loop to recover. That works, but when anything else goes wrong
the user is left looking at "Starting up..." with no way to tell a slow start
from a dead one.

Polling here instead means the browser opens only when there is something to
show, and the console window reports progress meanwhile.
"""

from __future__ import annotations

import sys
import time
import webbrowser
from http.client import HTTPConnection

TIMEOUT_S = 180.0
POLL_S = 0.25


def ready(port: int) -> bool:
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/api/health")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except Exception:  # noqa: BLE001 - not listening yet is the normal case
        return False


def main(argv: list[str]) -> int:
    port = int(argv[1]) if len(argv) > 1 else 8000
    url = "http://127.0.0.1:%d" % port

    started = time.time()
    announced = 0.0
    while time.time() - started < TIMEOUT_S:
        if ready(port):
            print("   Ready. Opening %s" % url)
            webbrowser.open(url)
            return 0
        waited = time.time() - started
        if waited - announced >= 5.0:
            announced = waited
            print("   Still starting... (%ds)" % int(waited))
        time.sleep(POLL_S)

    # Open it anyway: the frontend retries, and a visible page with a real
    # message beats a console the user is not reading.
    print("   Server did not answer within %ds. Opening %s anyway." % (TIMEOUT_S, url))
    webbrowser.open(url)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
