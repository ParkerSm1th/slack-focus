from __future__ import annotations

import html
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import LABEL_SCORES
from .storage import Store


class ReviewServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        port = self.httpd.server_address[1] if self.httpd else self.port
        return f"http://{self.host}:{port}"

    def start(self) -> str:
        if self.httpd:
            return self.url

        handler = _make_handler()
        try:
            self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        except OSError:
            self.httpd = ThreadingHTTPServer((self.host, 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self.url

    def open(self) -> str:
        url = self.start()
        webbrowser.open(url)
        return url

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/review"):
                self.send_error(404)
                return
            store = Store()
            try:
                messages = store.review_messages(limit=100)
                count = store.unlabeled_count()
            finally:
                store.close()

            body = _render(messages, count)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def do_POST(self) -> None:
            if self.path != "/label":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8")
            values = urllib.parse.parse_qs(payload)
            message_id = values.get("message_id", [""])[0]
            label = values.get("label", [""])[0]
            if not message_id or label not in LABEL_SCORES:
                self.send_error(400)
                return

            store = Store()
            try:
                store.set_label(message_id, label)
            finally:
                store.close()

            self.send_response(303)
            self.send_header("Location", "/review")
            self.end_headers()

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def _render(messages, unlabeled_count: int) -> str:
    rows = []
    for message in messages:
        score = "-" if message.score is None else f"{message.score:.3f}"
        level = message.level or "-"
        label = message.label or "unlabeled"
        link = (
            f'<a href="{html.escape(message.permalink)}" target="_blank">Open Slack</a>'
            if message.permalink
            else ""
        )
        buttons = " ".join(
            f"""
            <form method="post" action="/label">
              <input type="hidden" name="message_id" value="{html.escape(message.message_id)}">
              <input type="hidden" name="label" value="{label_name}">
              <button type="submit">{label_name}</button>
            </form>
            """
            for label_name in LABEL_SCORES
        )
        rows.append(
            f"""
            <article>
              <div class="meta">
                <strong>{html.escape(label)}</strong>
                <span>score {score}</span>
                <span>level {html.escape(level)}</span>
                <span>{html.escape(message.channel_name or message.channel_id or "")}</span>
                {link}
              </div>
              <p>{html.escape(message.text)}</p>
              <div class="actions">{buttons}</div>
            </article>
            """
        )

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Slack Priority Review</title>
      <style>
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0;
          background: #f7f7f5;
          color: #1d1d1f;
        }}
        header {{
          position: sticky;
          top: 0;
          background: #fff;
          border-bottom: 1px solid #ddd;
          padding: 14px 20px;
        }}
        main {{
          max-width: 980px;
          margin: 0 auto;
          padding: 20px;
        }}
        article {{
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 14px;
          margin-bottom: 12px;
        }}
        .meta {{
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
          font-size: 13px;
          color: #555;
        }}
        p {{
          white-space: pre-wrap;
          line-height: 1.4;
        }}
        .actions {{
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }}
        form {{
          display: inline;
        }}
        button {{
          border: 1px solid #bbb;
          background: #fff;
          border-radius: 6px;
          padding: 7px 10px;
          cursor: pointer;
        }}
        button:hover {{
          background: #ececec;
        }}
      </style>
    </head>
    <body>
      <header>
        <strong>Slack Priority Review</strong>
        <span>{unlabeled_count} unlabeled messages</span>
      </header>
      <main>
        {''.join(rows) if rows else '<p>No messages yet. Run backfill or start listening.</p>'}
      </main>
    </body>
    </html>
    """
