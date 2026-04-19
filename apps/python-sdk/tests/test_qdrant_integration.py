"""
Unit tests for the Qdrant crawl_to_store integration.

All external calls (Firecrawl API + Qdrant client) are mocked so no live
services are required.
"""

import unittest
from unittest.mock import MagicMock, patch

from firecrawl.v2.integrations.qdrant import (
    crawl_to_store,
    CrawlToStoreResult,
    _content_hash,
    _url_to_point_id,
    _url_for_doc,
    _get_content,
)
from firecrawl.v2.types import Document, DocumentMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(url: str, markdown: str = "hello world", title: str = "Test") -> Document:
    """Build a minimal Document for testing."""
    meta = DocumentMetadata(url=url, title=title)
    return Document(markdown=markdown, metadata=meta)


def _make_crawl_job(docs):
    """Build a mock CrawlJob-like object."""
    job = MagicMock()
    job.data = docs
    job.status = "completed"
    return job


def _make_qdrant_client(existing_point=None):
    """Return a mock QdrantClient."""
    qc = MagicMock()
    # scroll returns (results, next_page_offset)
    qc.scroll.return_value = ([existing_point] if existing_point else [], None)
    qc.upsert.return_value = MagicMock()
    return qc


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_content_hash_deterministic(self):
        h1 = _content_hash("hello")
        h2 = _content_hash("hello")
        self.assertEqual(h1, h2)

    def test_content_hash_differs_for_different_text(self):
        self.assertNotEqual(_content_hash("hello"), _content_hash("world"))

    def test_content_hash_is_64_hex_chars(self):
        h = _content_hash("test")
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_url_to_point_id_deterministic(self):
        id1 = _url_to_point_id("https://example.com/page")
        id2 = _url_to_point_id("https://example.com/page")
        self.assertEqual(id1, id2)

    def test_url_to_point_id_differs_for_different_urls(self):
        self.assertNotEqual(
            _url_to_point_id("https://example.com/a"),
            _url_to_point_id("https://example.com/b"),
        )

    def test_url_to_point_id_is_valid_uuid_string(self):
        import uuid
        result = _url_to_point_id("https://example.com")
        uuid.UUID(result)  # raises if invalid

    def test_url_for_doc_returns_metadata_url(self):
        doc = _make_doc("https://example.com/page")
        self.assertEqual(_url_for_doc(doc), "https://example.com/page")

    def test_url_for_doc_falls_back_to_source_url(self):
        meta = DocumentMetadata(source_url="https://example.com/src")
        doc = Document(markdown="x", metadata=meta)
        self.assertEqual(_url_for_doc(doc), "https://example.com/src")

    def test_url_for_doc_returns_none_without_metadata(self):
        doc = Document(markdown="x")
        self.assertIsNone(_url_for_doc(doc))

    def test_get_content_markdown(self):
        doc = _make_doc("https://x.com", markdown="# Hi")
        self.assertEqual(_get_content(doc, "markdown"), "# Hi")

    def test_get_content_html(self):
        doc = Document(html="<h1>Hi</h1>", metadata=DocumentMetadata(url="https://x.com"))
        self.assertEqual(_get_content(doc, "html"), "<h1>Hi</h1>")

    def test_get_content_missing_field_returns_none(self):
        doc = Document(metadata=DocumentMetadata(url="https://x.com"))
        self.assertIsNone(_get_content(doc, "markdown"))


# ---------------------------------------------------------------------------
# crawl_to_store — happy path: all pages are new
# ---------------------------------------------------------------------------

class TestCrawlToStoreNewPages(unittest.TestCase):

    def setUp(self):
        self.docs = [
            _make_doc("https://example.com/a", markdown="page a content"),
            _make_doc("https://example.com/b", markdown="page b content"),
        ]
        self.crawl_job = _make_crawl_job(self.docs)
        self.fc = MagicMock()
        self.fc.crawl.return_value = self.crawl_job
        self.qc = _make_qdrant_client(existing_point=None)  # no existing points
        self.embed = MagicMock(return_value=[0.1, 0.2, 0.3])

    def test_returns_crawl_to_store_result(self):
        result = crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
        )
        self.assertIsInstance(result, CrawlToStoreResult)

    def test_all_new_pages_upserted(self):
        result = crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
        )
        self.assertEqual(result.total, 2)
        self.assertEqual(result.upserted, 2)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.updated, 0)

    def test_embedding_fn_called_once_per_page(self):
        crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
        )
        self.assertEqual(self.embed.call_count, 2)

    def test_qdrant_upsert_called_for_each_page(self):
        crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
        )
        self.assertEqual(self.qc.upsert.call_count, 2)

    def test_point_payload_contains_url_and_hash(self):
        crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
        )
        first_call_kwargs = self.qc.upsert.call_args_list[0]
        point = first_call_kwargs[1]["points"][0]
        self.assertIn("url", point.payload)
        self.assertIn("content_hash", point.payload)

    def test_crawl_called_with_url_and_kwargs(self):
        crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
            limit=50,
        )
        self.fc.crawl.assert_called_once()
        call_kwargs = self.fc.crawl.call_args[1]
        self.assertEqual(call_kwargs["limit"], 50)

    def test_crawl_job_attached_to_result(self):
        result = crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
        )
        self.assertIs(result.crawl_job, self.crawl_job)


# ---------------------------------------------------------------------------
# crawl_to_store — skip_existing=True, identical content
# ---------------------------------------------------------------------------

class TestCrawlToStoreSkipExisting(unittest.TestCase):

    def setUp(self):
        self.doc = _make_doc("https://example.com/a", markdown="unchanged content")
        self.crawl_job = _make_crawl_job([self.doc])
        self.fc = MagicMock()
        self.fc.crawl.return_value = self.crawl_job
        self.embed = MagicMock(return_value=[0.1, 0.2, 0.3])

        # Build an existing Qdrant point with matching hash
        existing = MagicMock()
        existing.payload = {
            "url": "https://example.com/a",
            "content_hash": _content_hash("unchanged content"),
        }
        self.qc = _make_qdrant_client(existing_point=existing)

    def test_unchanged_page_is_skipped(self):
        result = crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
            skip_existing=True,
        )
        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.upserted, 0)

    def test_embedding_not_called_for_skipped_page(self):
        crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
            skip_existing=True,
        )
        self.embed.assert_not_called()

    def test_qdrant_not_upserted_for_skipped_page(self):
        crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
            skip_existing=True,
        )
        self.qc.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# crawl_to_store — skip_existing=True, changed content → update
# ---------------------------------------------------------------------------

class TestCrawlToStoreUpdateChanged(unittest.TestCase):

    def setUp(self):
        self.doc = _make_doc("https://example.com/a", markdown="NEW content")
        self.crawl_job = _make_crawl_job([self.doc])
        self.fc = MagicMock()
        self.fc.crawl.return_value = self.crawl_job
        self.embed = MagicMock(return_value=[0.5, 0.6])

        # Existing point has a DIFFERENT hash (stale content)
        existing = MagicMock()
        existing.payload = {
            "url": "https://example.com/a",
            "content_hash": _content_hash("OLD content"),
        }
        self.qc = _make_qdrant_client(existing_point=existing)

    def test_changed_page_is_updated(self):
        result = crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
            skip_existing=True,
        )
        self.assertEqual(result.upserted, 1)
        self.assertEqual(result.updated, 1)
        self.assertEqual(result.skipped, 0)

    def test_embedding_called_for_updated_page(self):
        crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
            skip_existing=True,
        )
        self.embed.assert_called_once()


# ---------------------------------------------------------------------------
# crawl_to_store — skip_existing=False always upserts
# ---------------------------------------------------------------------------

class TestCrawlToStoreNoSkip(unittest.TestCase):

    def setUp(self):
        self.doc = _make_doc("https://example.com/a", markdown="same content")
        self.crawl_job = _make_crawl_job([self.doc])
        self.fc = MagicMock()
        self.fc.crawl.return_value = self.crawl_job
        self.embed = MagicMock(return_value=[0.1])

        # Even though a point exists with same hash…
        existing = MagicMock()
        existing.payload = {"url": "https://example.com/a", "content_hash": _content_hash("same content")}
        self.qc = _make_qdrant_client(existing_point=existing)

    def test_page_upserted_even_if_unchanged(self):
        result = crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
            skip_existing=False,
        )
        self.assertEqual(result.upserted, 1)
        self.assertEqual(result.skipped, 0)

    def test_qdrant_scroll_not_called_when_skip_false(self):
        crawl_to_store(
            self.fc, "https://example.com",
            qdrant_client=self.qc,
            collection_name="test_col",
            embedding_fn=self.embed,
            skip_existing=False,
        )
        self.qc.scroll.assert_not_called()


# ---------------------------------------------------------------------------
# crawl_to_store — edge cases / failure paths
# ---------------------------------------------------------------------------

class TestCrawlToStoreEdgeCases(unittest.TestCase):

    def _base_fc(self, docs):
        fc = MagicMock()
        fc.crawl.return_value = _make_crawl_job(docs)
        return fc

    def test_empty_crawl_returns_zeros(self):
        fc = self._base_fc([])
        qc = _make_qdrant_client()
        result = crawl_to_store(
            fc, "https://example.com",
            qdrant_client=qc,
            collection_name="col",
            embedding_fn=MagicMock(),
        )
        self.assertEqual(result.total, 0)
        self.assertEqual(result.upserted, 0)
        self.assertEqual(result.skipped, 0)

    def test_doc_with_no_content_is_skipped(self):
        doc = Document(metadata=DocumentMetadata(url="https://x.com"))  # no markdown
        fc = self._base_fc([doc])
        qc = _make_qdrant_client()
        embed = MagicMock()
        result = crawl_to_store(
            fc, "https://example.com",
            qdrant_client=qc,
            collection_name="col",
            embedding_fn=embed,
        )
        self.assertEqual(result.skipped, 1)
        embed.assert_not_called()

    def test_invalid_content_key_raises_value_error(self):
        fc = self._base_fc([_make_doc("https://x.com")])
        with self.assertRaises(ValueError):
            crawl_to_store(
                fc, "https://example.com",
                qdrant_client=MagicMock(),
                collection_name="col",
                embedding_fn=MagicMock(),
                content_key="invalid_key",
            )

    def test_missing_qdrant_client_import_raises(self):
        """When qdrant_client package is not installed, a clear ImportError is raised."""
        doc = _make_doc("https://example.com/a")
        fc = self._base_fc([doc])
        qc = MagicMock()

        with patch.dict("sys.modules", {"qdrant_client": None, "qdrant_client.models": None}):
            with self.assertRaises((ImportError, TypeError)):
                crawl_to_store(
                    fc, "https://example.com",
                    qdrant_client=qc,
                    collection_name="col",
                    embedding_fn=MagicMock(return_value=[0.1]),
                )

    def test_embedding_error_recorded_not_raised(self):
        doc = _make_doc("https://example.com/a", markdown="content")
        fc = self._base_fc([doc])
        qc = _make_qdrant_client()
        embed = MagicMock(side_effect=RuntimeError("embed failed"))

        result = crawl_to_store(
            fc, "https://example.com",
            qdrant_client=qc,
            collection_name="col",
            embedding_fn=embed,
        )
        self.assertEqual(result.upserted, 0)
        self.assertIn("https://example.com/a", result.errors)

    def test_qdrant_upsert_error_recorded_not_raised(self):
        doc = _make_doc("https://example.com/a", markdown="content")
        fc = self._base_fc([doc])
        qc = _make_qdrant_client()
        qc.upsert.side_effect = Exception("connection refused")

        result = crawl_to_store(
            fc, "https://example.com",
            qdrant_client=qc,
            collection_name="col",
            embedding_fn=MagicMock(return_value=[0.1]),
        )
        self.assertEqual(result.upserted, 0)
        self.assertIn("https://example.com/a", result.errors)

    def test_deterministic_point_id_same_across_runs(self):
        """Re-crawling the same URL produces the same point ID so upsert is idempotent."""
        doc = _make_doc("https://example.com/stable")
        fc1 = self._base_fc([doc])
        fc2 = self._base_fc([_make_doc("https://example.com/stable", markdown="v2")])
        qc = _make_qdrant_client()
        embed = MagicMock(return_value=[0.1])

        crawl_to_store(fc1, "https://example.com", qdrant_client=qc, collection_name="c", embedding_fn=embed)
        crawl_to_store(fc2, "https://example.com", qdrant_client=qc, collection_name="c", embedding_fn=embed)

        ids = [
            qc.upsert.call_args_list[i][1]["points"][0].id
            for i in range(2)
        ]
        self.assertEqual(ids[0], ids[1])

    def test_html_content_key(self):
        doc = Document(html="<p>hi</p>", metadata=DocumentMetadata(url="https://x.com"))
        fc = self._base_fc([doc])
        qc = _make_qdrant_client()
        embed = MagicMock(return_value=[0.1])

        result = crawl_to_store(
            fc, "https://x.com",
            qdrant_client=qc,
            collection_name="col",
            embedding_fn=embed,
            content_key="html",
        )
        self.assertEqual(result.upserted, 1)
        embed.assert_called_once_with("<p>hi</p>")

    def test_crawl_kwargs_forwarded(self):
        doc = _make_doc("https://example.com/page")
        fc = self._base_fc([doc])
        qc = _make_qdrant_client()

        crawl_to_store(
            fc, "https://example.com",
            qdrant_client=qc,
            collection_name="col",
            embedding_fn=MagicMock(return_value=[0.1]),
            limit=10,
            max_discovery_depth=3,
            allow_subdomains=True,
        )

        call_kwargs = fc.crawl.call_args[1]
        self.assertEqual(call_kwargs["limit"], 10)
        self.assertEqual(call_kwargs["max_discovery_depth"], 3)
        self.assertTrue(call_kwargs["allow_subdomains"])


# ---------------------------------------------------------------------------
# FirecrawlClient integration — method is accessible on the client
# ---------------------------------------------------------------------------

class TestFirecrawlClientIntegration(unittest.TestCase):

    def test_crawl_to_store_on_firecrawl_client(self):
        """crawl_to_store is reachable via FirecrawlClient."""
        from firecrawl.v2.client import FirecrawlClient
        self.assertTrue(callable(FirecrawlClient.crawl_to_store))

    def test_crawl_to_store_on_unified_client(self):
        """crawl_to_store is reachable via the unified Firecrawl client."""
        from firecrawl import Firecrawl
        # Instantiate without API key (self-hosted mode)
        fc = Firecrawl(api_key="test", api_url="http://localhost:3002")
        self.assertTrue(callable(fc.crawl_to_store))

    def test_crawl_to_store_result_exported(self):
        """CrawlToStoreResult is importable from the top-level package."""
        from firecrawl import CrawlToStoreResult  # noqa: F401


if __name__ == "__main__":
    unittest.main()
