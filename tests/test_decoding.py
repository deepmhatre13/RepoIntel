import os
import tempfile
import pytest
from packages.parsing.decoding import decode_file_bytes, DecodedFileResult
from packages.parsing.pipeline import run_parsing_pipeline
from packages.parsing.chunking import CodeChunk
from packages.knowledge.persistence import upsert_file_result

# TEST 1 — ASCII
def test_ascii():
    raw = b"Django==5.0.6\n"
    res = decode_file_bytes(raw, "test.txt")
    assert res.success is True
    assert res.classification == "text"
    assert res.encoding in ("ascii", "utf-8")
    assert res.content == "Django==5.0.6\n"
    assert "\x00" not in res.content

# TEST 2 — UTF-8
def test_utf8():
    text_input = "Django==5.0.6\n🚀 features=enabled\n"
    raw = text_input.encode("utf-8")
    res = decode_file_bytes(raw, "test.txt")
    assert res.success is True
    assert res.classification == "text"
    assert res.encoding == "utf-8"
    assert res.content == text_input
    assert "\x00" not in res.content

# TEST 3 — UTF-8 BOM
def test_utf8_bom():
    text_input = "Django==5.0.6\ngunicorn\n"
    raw = b"\xef\xbb\xbf" + text_input.encode("utf-8")
    res = decode_file_bytes(raw, "test.txt")
    assert res.success is True
    assert res.classification == "text"
    assert res.encoding == "utf-8-sig"
    assert res.content == text_input
    assert "\x00" not in res.content

# TEST 4 — UTF-16 LE BOM
def test_utf16_le_bom():
    text_input = "Django==5.0.6\ngunicorn\n"
    raw = b"\xff\xfe" + text_input.encode("utf-16-le")
    res = decode_file_bytes(raw, "requirements.txt")
    assert res.success is True
    assert res.classification == "text"
    assert res.encoding == "utf-16-le"
    assert res.content == text_input
    assert "\x00" not in res.content

# TEST 5 — UTF-16 BE BOM
def test_utf16_be_bom():
    text_input = "Django==5.0.6\ngunicorn\n"
    raw = b"\xfe\xff" + text_input.encode("utf-16-be")
    res = decode_file_bytes(raw, "test.txt")
    assert res.success is True
    assert res.classification == "text"
    assert res.encoding == "utf-16-be"
    assert res.content == text_input
    assert "\x00" not in res.content

# TEST 6 — Requirements-like UTF-16 file through ingestion pipeline
def test_requirements_utf16_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        req_path = os.path.join(tmpdir, "requirements.txt")
        text_input = "Django==5.0.6\ngunicorn==21.2.0\n"
        with open(req_path, "wb") as f:
            f.write(b"\xff\xfe" + text_input.encode("utf-16-le"))

        pipeline_res = run_parsing_pipeline("repo-test-1", tmpdir, ["requirements.txt"])
        assert pipeline_res.files_parsed == 1
        assert len(pipeline_res.failures) == 0
        found_django = any("Django==5.0.6" in c.content for c in pipeline_res.chunks)
        assert found_django is True
        for chunk in pipeline_res.chunks:
            assert "\x00" not in chunk.content

# TEST 7 — Genuine binary file
def test_genuine_binary_file():
    raw = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\x00\x00\x00\x1f\xf3\xff"
    res = decode_file_bytes(raw, "image.png")
    assert res.success is False
    assert res.classification == "binary"
    assert "\x00" not in res.content

# TEST 8 — Malformed text
def test_malformed_text():
    raw = b"\x80\x81\x82\x83\x00\x84\x85"
    res = decode_file_bytes(raw, "corrupt.bin")
    assert res.success is False
    assert res.classification in ("binary", "undecodable")
    assert "\x00" not in res.content

# TEST 9 — Database boundary (safety invariant check)
@pytest.mark.asyncio
async def test_database_boundary_invariant():
    class DummySession:
        pass

    bad_chunk = CodeChunk(
        repo_id="test-repo",
        file_path="bad.py",
        symbol_name="bad_func",
        line_start=1,
        line_end=5,
        content="def bad_func():\n    return 'D\x00j\x00a\x00n\x00g\x00o'\n",
        content_hash="dummyhash",
        language="Python",
    )

    with pytest.raises(ValueError, match="Database safety invariant violation"):
        await upsert_file_result(
            session=DummySession(),
            repo_id="test-repo",
            file_path="bad.py",
            language="Python",
            content_hash="dummyhash",
            size_bytes=100,
            symbols=[],
            edges=[],
            chunks=[bad_chunk],
        )

# TEST 10 — Existing normal repository ingestion
def test_normal_repository_ingestion():
    with tempfile.TemporaryDirectory() as tmpdir:
        py_path = os.path.join(tmpdir, "main.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write("def hello():\n    return 'world'\n")

        pipeline_res = run_parsing_pipeline("repo-test-2", tmpdir, ["main.py"])
        assert pipeline_res.files_parsed == 1
        assert len(pipeline_res.failures) == 0
        assert len(pipeline_res.symbols) > 0

# TEST 11 — Real Aiverse file test if cloned
def test_real_aiverse_requirements():
    aiverse_req = os.path.join("scratch", "aiverse_test", "aiverse_backend", "backend", "ml", "requirements.txt")
    if os.path.exists(aiverse_req):
        with open(aiverse_req, "rb") as f:
            raw = f.read()
        res = decode_file_bytes(raw, aiverse_req)
        assert res.success is True
        assert res.encoding == "utf-16-le"
        assert res.classification == "text"
        assert "\x00" not in res.content
        assert "Django==5.0.6" in res.content
