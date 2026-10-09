import secrets


def enclose_repository_evidence(content: str) -> str:
    """Wrap repository-controlled text in a nonce-bounded untrusted-data block."""
    boundary = secrets.token_hex(16)
    return (
        f"BEGIN UNTRUSTED REPOSITORY DATA {boundary}\n"
        f"{content}\n"
        f"END UNTRUSTED REPOSITORY DATA {boundary}"
    )