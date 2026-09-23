"""Tests for the dictionaries SDK surface.

Exercises the full CRUD + bulk-replace flow against a stubbed
``client._request`` so the SDK shape is verified without a live server.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import patch

from kugelaudio import (
    BulkReplaceResult,
    Dictionary,
    DictionaryEntry,
    DictionaryEntryList,
    KugelAudio,
)


def _make_client() -> KugelAudio:
    return KugelAudio(api_key="test_key")


def _dict_row(id_: int = 1, project_id: int = 42, **overrides: Any) -> Dict[str, Any]:
    row = {
        "id": id_,
        "project_id": project_id,
        "name": "Brand names",
        "description": None,
        "language": None,
        "is_active": True,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def _entry_row(id_: int, dictionary_id: int = 1, **overrides: Any) -> Dict[str, Any]:
    row = {
        "id": id_,
        "dictionary_id": dictionary_id,
        "word": "Kubernetes",
        "replacement": "koo-ber-net-eez",
        "ipa": None,
        "case_sensitive": False,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    row.update(overrides)
    return row


class TestDictionariesCrud:
    def test_list(self):
        client = _make_client()
        with patch.object(
            client,
            "_request",
            return_value={"dictionaries": [_dict_row(1), _dict_row(2, name="Other")]},
        ) as m:
            result = client.dictionaries.list()
        m.assert_called_once_with("GET", "/v1/dictionaries", params=None)
        assert len(result) == 2
        assert isinstance(result[0], Dictionary)
        assert result[0].name == "Brand names"


    def test_create(self):
        client = _make_client()
        with patch.object(
            client,
            "_request",
            return_value=_dict_row(7, name="Glossary", description="hi", language="en"),
        ) as m:
            d = client.dictionaries.create(
                name="Glossary", description="hi", language="en"
            )
        m.assert_called_once_with(
            "POST",
            "/v1/dictionaries",
            params=None,
            json_data={
                "name": "Glossary",
                "description": "hi",
                "language": "en",
            },
        )
        assert d.id == 7
        assert d.name == "Glossary"


    def test_update_sends_only_provided_fields(self):
        client = _make_client()
        with patch.object(
            client,
            "_request",
            return_value=_dict_row(1, is_active=False),
        ) as m:
            client.dictionaries.update(1, is_active=False)
        m.assert_called_once_with(
            "PATCH",
            "/v1/dictionaries/1",
            params=None,
            json_data={"is_active": False},
        )


class TestEntriesCrud:
    def test_list_with_search(self):
        client = _make_client()
        with patch.object(
            client,
            "_request",
            return_value={
                "entries": [_entry_row(1)],
                "total": 1,
                "limit": 50,
                "offset": 0,
            },
        ) as m:
            res = client.dictionaries.entries.list(1, search="kub", limit=50)
        m.assert_called_once_with(
            "GET",
            "/v1/dictionaries/1/entries",
            params={"limit": 50, "offset": 0, "search": "kub"},
        )
        assert isinstance(res, DictionaryEntryList)
        assert res.total == 1
        assert res.entries[0].word == "Kubernetes"

    def test_add(self):
        client = _make_client()
        with patch.object(
            client,
            "_request",
            return_value=_entry_row(11, word="Postgres", replacement="post-gres"),
        ) as m:
            e = client.dictionaries.entries.add(
                1, word="Postgres", replacement="post-gres"
            )
        m.assert_called_once_with(
            "POST",
            "/v1/dictionaries/1/entries",
            params=None,
            json_data={
                "word": "Postgres",
                "replacement": "post-gres",
                "case_sensitive": False,
            },
        )
        assert isinstance(e, DictionaryEntry)


    def test_replace_all(self):
        client = _make_client()
        payload = [
            {"word": "Postgres", "replacement": "post-gres"},
            {"word": "K8s", "replacement": "kubernetes"},
        ]
        with patch.object(
            client,
            "_request",
            return_value={"upserted": 2, "deleted": 3, "total": 2},
        ) as m:
            result = client.dictionaries.entries.replace_all(1, entries=payload)
        m.assert_called_once_with(
            "PUT",
            "/v1/dictionaries/1/entries",
            params=None,
            json_data={"entries": payload},
        )
        assert isinstance(result, BulkReplaceResult)
        assert result.upserted == 2
        assert result.deleted == 3
