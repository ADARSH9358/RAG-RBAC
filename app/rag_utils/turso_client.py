"""
Turso HTTP API client that mimics the sqlite3 connection/cursor interface.
Uses Turso's /v2/pipeline REST endpoint — no Rust/compilation needed.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

_DB_URL   = os.getenv("SQLITE_DB_PATH", "")   # e.g. libsql://rolesdocs-trivedi.aws-ap-south-1.turso.io
_DB_TOKEN = os.getenv("SQLITE_TOKEN", "")

# Convert libsql:// scheme → https://
def _http_url(url: str) -> str:
    return url.replace("libsql://", "https://").rstrip("/")

TURSO_HTTP_URL = _http_url(_DB_URL) + "/v2/pipeline"


class TursoCursor:
    def __init__(self, conn: "TursoConnection"):
        self._conn = conn
        self.description = None
        self.rowcount = -1
        self._rows: list = []

    def execute(self, sql: str, params=()):
        # Convert ? placeholders to named args Turso expects
        args = [{"type": _infer_type(p), "value": str(p) if p is not None else None} for p in params]
        payload = {
            "requests": [
                {"type": "execute", "stmt": {"sql": sql, "args": args}},
                {"type": "close"},
            ]
        }
        headers = {
            "Authorization": f"Bearer {_DB_TOKEN}",
            "Content-Type": "application/json",
        }
        resp = requests.post(TURSO_HTTP_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        result = data["results"][0]
        if result["type"] == "error":
            raise Exception(result["error"]["message"])

        inner = result.get("response", {}).get("result", {})
        cols = inner.get("cols", [])
        rows = inner.get("rows", [])

        self.description = [(c["name"], None, None, None, None, None, None) for c in cols]
        self.rowcount = inner.get("affected_row_count", -1)
        self._rows = [
            tuple(cell.get("value") for cell in row)
            for row in rows
        ]

    def executescript(self, script: str):
        # Split on semicolons and execute each statement
        stmts = [s.strip() for s in script.split(";") if s.strip()]
        for stmt in stmts:
            self.execute(stmt)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class TursoConnection:
    def __init__(self):
        pass

    def cursor(self) -> TursoCursor:
        return TursoCursor(self)

    def execute(self, sql: str, params=()):
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def executescript(self, script: str):
        cur = self.cursor()
        cur.executescript(script)

    def commit(self):
        pass  # Turso auto-commits

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def connect(url: str = None, auth_token: str = None) -> TursoConnection:
    """Drop-in replacement for sqlite3.connect() / libsql.connect()."""
    return TursoConnection()


def _infer_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    return "text"
