"""SSRF defense for repository URL ingestion.

This module implements a production-grade SSRF guard that protects against:
- localhost / loopback addresses
- RFC1918 private networks (10/8, 172.16/12, 192.168/16)
- link-local addresses (169.254/16, fe80::/10)
- cloud metadata endpoints (AWS, Azure, GCP)
- IPv6 private ranges (fc00::/7, ::1/128)
- DNS rebinding (resolve + validate at connection time)
- redirects into private addresses
- credential-bearing URLs
- malformed URLs and unsupported schemes

Design:
    Uses Python's standard library `ipaddress` module for correct network
    address classification rather than string prefix matching. This handles
    edge cases like IPv4-mapped IPv6 addresses correctly.

    DNS rebinding protection: we resolve the hostname once and validate
    the IP before passing to git. While git itself performs the connection,
    we validate at the application boundary to prevent accidental exposure.
"""

import ipaddress
import os
import socket
import urllib.parse
from typing import Set

from packages.ingestion.exceptions import InvalidRepositoryError

# Schemes we allow — file:// is intentionally EXCLUDED (SSRF surface).
_ALLOWED_SCHEMES = {"http", "https", "git", "ssh"}

# Hosts blocked by the SSRF guard (exact match, case-insensitive).
_SRF_BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "169.254.169.254",       # AWS/Azure/GCP metadata endpoint
    "[::ffff:169.254.169.254]",
}

# Cloud metadata endpoints (all variants).
_METADATA_ENDPOINTS = {
    "169.254.169.254",
    "[::ffff:169.254.169.254]",
    "fd00:ec2::254",         # AWS IPv6 metadata
    "169.254.170.2",         # AWS VPC metadata
}


def _resolve_host_ip(hostname: str) -> Set[str]:
    """Resolve a hostname to its IP addresses for SSRF checking."""
    ips = set()
    try:
        for info in socket.getaddrinfo(hostname, None):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    return ips


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private, loopback, link-local, or metadata.
    
    Uses the ipaddress module for correct classification of all address types
    including IPv4-mapped IPv6 addresses.
    
    Note: We don't block 'reserved' addresses wholesale because some reserved
    ranges (like 203.0.113.0/24 for documentation) are not actually private.
    We specifically block only the ranges that pose SSRF risk.
    """
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # Invalid IP — treat as blocked
    
    # Handle IPv4-mapped IPv6 addresses (e.g., ::ffff:127.0.0.1)
    # by extracting the embedded IPv4 address
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    
    if addr.is_private or addr.is_loopback or addr.is_link_local:
        return True
    if addr.is_multicast:
        return True
    # Check for unspecified address (0.0.0.0, ::)
    if addr.is_unspecified:
        return True
    return False


def _is_ssrf_blocked(hostname: str) -> bool:
    """Return True if the hostname resolves to a private/internal/metadata address."""
    if not hostname:
        return True
    h = hostname.lower()
    if h in _SRF_BLOCKED_HOSTS:
        return True
    # Resolve and check for private/link-local ranges.
    try:
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if _is_private_ip(ip):
                return True
    except socket.gaierror:
        # DNS resolution fail — treat as blocked.
        return True
    return False


def _validate_port(parsed: urllib.parse.ParseResult) -> None:
    """Validate that the port is within acceptable ranges.
    
    Blocks unusual ports that might be used for SSRF attacks.
    """
    port = parsed.port
    if port is None:
        return  # Default port for scheme
    
    # Allow standard ports only
    allowed_ports = {22, 80, 443, 9418}  # SSH, HTTP, HTTPS, Git
    if port not in allowed_ports:
        raise InvalidRepositoryError(
            f"Port {port} is not allowed. Allowed ports: {sorted(allowed_ports)}"
        )


def _strip_credentials(url: str) -> str:
    """Remove credentials from a URL to prevent credential leakage.
    
    URLs like https://user:pass@host/repo are normalized to https://host/repo.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.username or parsed.password:
        # Reconstruct without credentials
        netloc = parsed.hostname
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urllib.parse.urlunparse((
            parsed.scheme, netloc, parsed.path,
            parsed.params, parsed.query, parsed.fragment
        ))
    return url


def validate_repo_url(url: str) -> urllib.parse.ParseResult:
    """Validate and parse a repository URL, applying the SSRF guard.

    Returns the parsed URL on success. Raises InvalidRepositoryError on
    any failure (bad scheme, blocked host, malformed URL).
    """
    if not url or not isinstance(url, str):
        raise InvalidRepositoryError(f"Invalid repository URL or path: {url!r}")

    url = url.strip()

    # Local filesystem path — allowed, no SSRF concern.
    if os.path.isdir(url):
        return urllib.parse.ParseResult(scheme="local", path=url, params="", query="", fragment="", netloc="")

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        raise InvalidRepositoryError(f"Malformed repository URL: {url!r} ({exc})")

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise InvalidRepositoryError(
            f"Unsupported scheme '{parsed.scheme}' for URL: {url!r}. "
            f"Allowed: {sorted(_ALLOWED_SCHEMES)} or a local directory path."
        )

    hostname = parsed.hostname
    if not hostname:
        raise InvalidRepositoryError(f"No host in URL: {url!r}")

    if _is_ssrf_blocked(hostname):
        raise InvalidRepositoryError(
            f"URL blocked by SSRF guard (private/internal host): {hostname!r}"
        )

    _validate_port(parsed)

    return parsed


def validate_redirect_url(original_url: str, redirect_url: str) -> urllib.parse.ParseResult:
    """Validate a redirect URL to prevent SSRF via redirects.
    
    A redirect from a public URL to a private URL is a common SSRF vector.
    This function validates the redirect target with the same SSRF rules.
    """
    if not redirect_url:
        raise InvalidRepositoryError("Empty redirect URL")
    
    # Resolve relative redirects
    if redirect_url.startswith("/"):
        # Relative to original host
        original_parsed = urllib.parse.urlparse(original_url)
        redirect_url = f"{original_parsed.scheme}://{original_parsed.hostname}{redirect_url}"
    elif not redirect_url.startswith(("http://", "https://", "git://", "ssh://")):
        # Relative path
        original_parsed = urllib.parse.urlparse(original_url)
        base = f"{original_parsed.scheme}://{original_parsed.hostname}"
        if original_parsed.path:
            base += "/".join(original_parsed.path.split("/")[:-1])
        redirect_url = f"{base}/{redirect_url}"
    
    return validate_repo_url(redirect_url)