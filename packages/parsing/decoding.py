from dataclasses import dataclass
from typing import Optional

@dataclass
class DecodedFileResult:
    success: bool
    content: str = ""
    encoding: Optional[str] = None
    reason: Optional[str] = None
    classification: str = "text"

def decode_file_bytes(raw_bytes: bytes, file_path: str = "") -> DecodedFileResult:
    if not raw_bytes:
        return DecodedFileResult(success=True, content="", encoding="utf-8", classification="text")

    # 1. Check for standard BOMs (check 4-byte BOMs first, then 3-byte, then 2-byte)
    if raw_bytes.startswith(b"\xff\xfe\x00\x00"):
        try:
            text = raw_bytes[4:].decode("utf-32-le")
            if "\x00" in text:
                return DecodedFileResult(success=False, content="", encoding="utf-32-le", reason="Decoded text contains NUL bytes", classification="undecodable")
            return DecodedFileResult(success=True, content=text, encoding="utf-32-le", classification="text")
        except Exception:
            pass

    if raw_bytes.startswith(b"\x00\x00\xfe\xff"):
        try:
            text = raw_bytes[4:].decode("utf-32-be")
            if "\x00" in text:
                return DecodedFileResult(success=False, content="", encoding="utf-32-be", reason="Decoded text contains NUL bytes", classification="undecodable")
            return DecodedFileResult(success=True, content=text, encoding="utf-32-be", classification="text")
        except Exception:
            pass

    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw_bytes[3:].decode("utf-8")
            if "\x00" in text:
                return DecodedFileResult(success=False, content="", encoding="utf-8-sig", reason="Decoded text contains NUL bytes", classification="undecodable")
            return DecodedFileResult(success=True, content=text, encoding="utf-8-sig", classification="text")
        except Exception:
            pass

    if raw_bytes.startswith(b"\xff\xfe"):
        try:
            text = raw_bytes[2:].decode("utf-16-le")
            if "\x00" in text:
                return DecodedFileResult(success=False, content="", encoding="utf-16-le", reason="Decoded text contains NUL bytes", classification="undecodable")
            return DecodedFileResult(success=True, content=text, encoding="utf-16-le", classification="text")
        except Exception:
            pass

    if raw_bytes.startswith(b"\xfe\xff"):
        try:
            text = raw_bytes[2:].decode("utf-16-be")
            if "\x00" in text:
                return DecodedFileResult(success=False, content="", encoding="utf-16-be", reason="Decoded text contains NUL bytes", classification="undecodable")
            return DecodedFileResult(success=True, content=text, encoding="utf-16-be", classification="text")
        except Exception:
            pass

    # 2. Strict UTF-8 check without BOM
    try:
        text = raw_bytes.decode("utf-8")
        if "\x00" in text:
            if b"\x00" in raw_bytes:
                return DecodedFileResult(success=False, content="", encoding="utf-8", reason="Binary file containing NUL bytes", classification="binary")
            return DecodedFileResult(success=False, content="", encoding="utf-8", reason="UTF-8 decoded text contains NUL bytes", classification="undecodable")
        enc_name = "ascii" if all(b < 128 for b in raw_bytes) else "utf-8"
        return DecodedFileResult(success=True, content=text, encoding=enc_name, classification="text")
    except UnicodeDecodeError:
        pass

    # 3. Check UTF-16 without BOM (e.g. alternating NUL bytes)
    if len(raw_bytes) % 2 == 0:
        try:
            text = raw_bytes.decode("utf-16-le")
            if "\x00" not in text and len(text) > 0:
                printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
                if printable / len(text) > 0.85:
                    return DecodedFileResult(success=True, content=text, encoding="utf-16-le", classification="text")
        except Exception:
            pass

        try:
            text = raw_bytes.decode("utf-16-be")
            if "\x00" not in text and len(text) > 0:
                printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
                if printable / len(text) > 0.85:
                    return DecodedFileResult(success=True, content=text, encoding="utf-16-be", classification="text")
        except Exception:
            pass

    # 4. Fallback charset detection using charset_normalizer if available
    try:
        import charset_normalizer
        best = charset_normalizer.from_bytes(raw_bytes).best()
        if best is not None and getattr(best, "confidence", 0) >= 0.7:
            text = str(best)
            if "\x00" not in text and len(text) > 0:
                printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
                if printable / len(text) > 0.8:
                    return DecodedFileResult(
                        success=True,
                        content=text,
                        encoding=best.encoding,
                        classification="text"
                    )
    except Exception:
        pass

    # 5. Binary detection & Latin-1 fallback
    if b"\x00" in raw_bytes:
        return DecodedFileResult(
            success=False,
            content="",
            reason="Binary content detected (contains NUL bytes)",
            classification="binary"
        )

    try:
        text = raw_bytes.decode("latin-1")
        if "\x00" not in text:
            printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
            if len(text) > 0 and (printable / len(text) > 0.85):
                return DecodedFileResult(success=True, content=text, encoding="latin-1", classification="text")
            return DecodedFileResult(
                success=False,
                content="",
                reason="Binary content detected (high ratio of unprintable control characters)",
                classification="binary"
            )
    except Exception:
        pass

    return DecodedFileResult(
        success=False,
        content="",
        reason="Unable to decode file as valid text",
        classification="undecodable"
    )
