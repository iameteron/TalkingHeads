"""Static file server for the play_web client with caching disabled.

Plain ``python -m http.server`` does not send ``Cache-Control`` headers, so
browsers apply heuristic caching and may keep serving a stale ``index.js`` /
``index.html`` after the files change. This server forces revalidation so the
latest client assets always load.
"""

from __future__ import annotations

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8089
    httpd = ThreadingHTTPServer((host, port), NoCacheHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
