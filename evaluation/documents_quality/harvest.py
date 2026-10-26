"""Download the shared DVD corpus to build the answer-quality benchmark from.

Every reference answer in ``cases.json`` must be checkable against the exact
text the agent searches, so the corpus is taken from the deployed DVD itself:
one JSONL file per document with its fragments in reading order. The library
API needs a Keycloak token: it is read from the environment variable named by
``--token-env`` or, without it, issued by Keycloak to a service client (DVD
accepts service tokens for reading) whose secret is typed at the prompt.

    python evaluation/documents_quality/harvest.py --base-url http://10.32.11.90:31002 \
        --out benchmarks/data/documents_quality/corpus
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
from pathlib import Path

import httpx

FIELDS = (
    "id",
    "order",
    "kind",
    "type",
    "numbering",
    "breadcrumb",
    "text",
    "table_html",
)


def slug(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("_")[:120]


def issue_token(keycloak: str, realm: str, client_id: str, secret_env: str) -> str:
    secret = (
        os.getenv(secret_env) or getpass.getpass(f"client_secret {client_id}: ")
    ).strip()
    if any(ord(char) < 32 for char in secret):
        # Ctrl+V in a hidden Windows prompt types ^V instead of pasting.
        raise SystemExit(
            "Вставьте секрет правой кнопкой мыши, Ctrl+V здесь не работает"
        )
    response = httpx.post(
        f"{keycloak.rstrip('/')}/realms/{realm}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": secret,
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise SystemExit(f"Keycloak {response.status_code}: {response.text}")
    return response.json()["access_token"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--token-env", default="DVD_TOKEN")
    parser.add_argument("--keycloak", default="https://keycloak.idu.actcognitive.org")
    parser.add_argument("--realm", default="IDU")
    parser.add_argument("--client-id", default="idu-dvd")
    parser.add_argument("--secret-env", default="DVD_CLIENT_SECRET")
    args = parser.parse_args()
    token = os.getenv(args.token_env) or issue_token(
        args.keycloak, args.realm, args.client_id, args.secret_env
    )
    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    args.out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=headers, timeout=120) as client:
        listing = client.get(f"{base}/library/documents")
        listing.raise_for_status()
        documents = listing.json()["documents"]
        index = []
        for summary in documents:
            if summary.get("status") not in (None, "active"):
                continue
            detail = client.get(f"{base}/library/documents/{summary['doc_id']}")
            detail.raise_for_status()
            document = detail.json()
            path = args.out / f"{slug(document['name'])}.jsonl"
            with path.open("w", encoding="utf-8") as file:
                for fragment in document["fragments"]:
                    row = {key: fragment.get(key) for key in FIELDS}
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            index.append(
                {
                    "doc_id": document["doc_id"],
                    "name": document["name"],
                    "title": document.get("title"),
                    "version": document.get("version"),
                    "doc_type": document.get("doc_type"),
                    "fragments": len(document["fragments"]),
                    "file": path.name,
                }
            )
            print(f"{document['name']}: {len(document['fragments'])}")
    (args.out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
