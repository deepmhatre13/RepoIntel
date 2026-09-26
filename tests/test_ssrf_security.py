"""SSRF security tests for the production-grade SSRF guard.

Tests the enhanced security module that uses ipaddress for correct
network classification, plus port validation and redirect protection.
"""

import pytest
from packages.ingestion.security import (
    _is_private_ip,
    _is_ssrf_blocked,
    _validate_port,
    _strip_credentials,
    validate_repo_url,
    validate_redirect_url,
)
from packages.ingestion.exceptions import InvalidRepositoryError


class TestPrivateIPClassification:
    """Test IP address classification using the ipaddress module."""

    def test_loopback_ipv4(self):
        assert _is_private_ip("127.0.0.1") is True
        assert _is_private_ip("127.0.0.255") is True
        assert _is_private_ip("127.255.255.255") is True

    def test_loopback_ipv6(self):
        assert _is_private_ip("::1") is True

    def test_rfc1918_10_8(self):
        assert _is_private_ip("10.0.0.1") is True
        assert _is_private_ip("10.255.255.255") is True

    def test_rfc1918_172_16_12(self):
        assert _is_private_ip("172.16.0.1") is True
        assert _is_private_ip("172.31.255.255") is True

    def test_rfc1918_192_168_16(self):
        assert _is_private_ip("192.168.0.1") is True
        assert _is_private_ip("192.168.255.255") is True

    def test_link_local_ipv4(self):
        assert _is_private_ip("169.254.0.1") is True
        assert _is_private_ip("169.254.169.254") is True

    def test_link_local_ipv6(self):
        assert _is_private_ip("fe80::1") is True

    def test_unique_local_ipv6(self):
        assert _is_private_ip("fc00::1") is True
        assert _is_private_ip("fd00::1") is True

    def test_unspecified_addresses(self):
        assert _is_private_ip("0.0.0.0") is True
        assert _is_private_ip("::") is True

    def test_multicast_addresses(self):
        assert _is_private_ip("224.0.0.1") is True
        assert _is_private_ip("ff02::1") is True

    def test_invalid_ip(self):
        assert _is_private_ip("not-an-ip") is True
        assert _is_private_ip("") is True

    def test_public_ip_allowed(self):
        assert _is_private_ip("8.8.8.8") is False
        assert _is_private_ip("1.1.1.1") is False
        assert _is_private_ip("44.224.1.1") is False  # AWS public IP

    def test_public_ipv6_allowed(self):
        assert _is_private_ip("2001:4860:4860::8888") is False  # Google DNS

    def test_ipv4_mapped_ipv6(self):
        """IPv4-mapped IPv6 addresses should be classified correctly."""
        assert _is_private_ip("::ffff:127.0.0.1") is True
        assert _is_private_ip("::ffff:10.0.0.1") is True
        assert _is_private_ip("::ffff:8.8.8.8") is False


class TestPortValidation:
    """Test port validation."""

    def test_default_port_allowed(self):
        from urllib.parse import urlparse
        parsed = urlparse("https://github.com/repo")
        _validate_port(parsed)  # Should not raise

    def test_standard_ports_allowed(self):
        from urllib.parse import urlparse
        for port in [22, 80, 443, 9418]:
            parsed = urlparse(f"https://github.com:{port}/repo")
            _validate_port(parsed)  # Should not raise

    def test_unusual_port_blocked(self):
        from urllib.parse import urlparse
        parsed = urlparse("https://github.com:8080/repo")
        with pytest.raises(InvalidRepositoryError, match="not allowed"):
            _validate_port(parsed)

    def test_high_port_blocked(self):
        from urllib.parse import urlparse
        parsed = urlparse("https://github.com:49152/repo")
        with pytest.raises(InvalidRepositoryError, match="not allowed"):
            _validate_port(parsed)


class TestCredentialStripping:
    """Test credential stripping from URLs."""

    def test_strip_user_password(self):
        result = _strip_credentials("https://user:pass@github.com/repo")
        assert "user" not in result
        assert "pass" not in result
        assert "github.com/repo" in result

    def test_strip_token(self):
        result = _strip_credentials("https://ghp_xxxxxxxxxxxx@github.com/repo")
        assert "ghp_" not in result
        assert "github.com/repo" in result

    def test_no_credentials_unchanged(self):
        url = "https://github.com/repo"
        assert _strip_credentials(url) == url


class TestRedirectValidation:
    """Test redirect URL validation."""

    def test_relative_redirect_same_host(self):
        result = validate_redirect_url(
            "https://github.com/owner/repo",
            "/other-repo"
        )
        assert result.hostname == "github.com"

    def test_relative_path_redirect(self):
        result = validate_redirect_url(
            "https://github.com/owner/repo",
            "../other-repo"
        )
        assert result.hostname == "github.com"

    def test_absolute_redirect_to_public(self):
        result = validate_redirect_url(
            "https://github.com/owner/repo",
            "https://gitlab.com/owner/repo"
        )
        assert result.hostname == "gitlab.com"

    def test_redirect_to_private_blocked(self):
        with pytest.raises(InvalidRepositoryError, match="SSRF"):
            validate_redirect_url(
                "https://github.com/owner/repo",
                "http://127.0.0.1/admin"
            )

    def test_empty_redirect_blocked(self):
        with pytest.raises(InvalidRepositoryError, match="Empty"):
            validate_redirect_url("https://github.com/repo", "")
