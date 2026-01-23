"""Security utilities for Ultimate MCP Server."""

import base64
import hashlib
import hmac
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

from ultimate_mcp_server.config import get_env
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


def mask_api_key(api_key: str) -> str:
    """Mask API key for safe logging.

    Args:
        api_key: API key to mask

    Returns:
        Masked API key
    """
    if not api_key:
        return ""

    # Keep first 4 and last 4 characters, mask the rest
    if len(api_key) <= 8:
        return "*" * len(api_key)

    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


def validate_api_key(api_key: str, provider: str) -> bool:
    """Validate API key format for a provider.

    Args:
        api_key: API key to validate
        provider: Provider name

    Returns:
        True if API key format is valid
    """
    if not api_key:
        return False

    # Provider-specific validation patterns
    patterns = {
        "openai": r"^sk-[a-zA-Z0-9]{48}$",
        "anthropic": r"^sk-ant-[a-zA-Z0-9]{48}$",
        "deepseek": r"^sk-[a-zA-Z0-9]{32,64}$",
        "gemini": r"^[a-zA-Z0-9_-]{39}$",
        # Add more providers as needed
    }

    # Get pattern for provider
    pattern = patterns.get(provider.lower())
    if not pattern:
        # For unknown providers, check minimum length
        return len(api_key) >= 16

    # Check if API key matches the pattern
    return bool(re.match(pattern, api_key))


def generate_random_string(length: int = 32) -> str:
    """Generate a cryptographically secure random string.

    Args:
        length: Length of the string

    Returns:
        Random string
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(length)

    # Convert to URL-safe base64
    random_string = base64.urlsafe_b64encode(random_bytes).decode("utf-8")

    # Truncate to desired length
    return random_string[:length]


def generate_api_key(prefix: str = "lgw") -> str:
    """Generate an API key for the gateway.

    Args:
        prefix: Key prefix

    Returns:
        Generated API key
    """
    # Generate timestamp
    timestamp = int(time.time())

    # Generate random bytes
    random_bytes = secrets.token_bytes(24)

    # Combine and encode
    timestamp_bytes = timestamp.to_bytes(4, byteorder="big")
    combined = timestamp_bytes + random_bytes
    encoded = base64.urlsafe_b64encode(combined).decode("utf-8").rstrip("=")

    # Add prefix
    return f"{prefix}-{encoded}"


def create_hmac_signature(key: str, message: str, algorithm: str = "sha256") -> str:
    """Create an HMAC signature.

    Args:
        key: Secret key
        message: Message to sign
        algorithm: Hash algorithm to use

    Returns:
        HMAC signature as hexadecimal string
    """
    # Convert inputs to bytes
    key_bytes = key.encode("utf-8")
    message_bytes = message.encode("utf-8")

    # Create HMAC
    if algorithm == "sha256":
        h = hmac.new(key_bytes, message_bytes, hashlib.sha256)
    elif algorithm == "sha512":
        h = hmac.new(key_bytes, message_bytes, hashlib.sha512)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Return hexadecimal digest
    return h.hexdigest()


def verify_hmac_signature(
    key: str, message: str, signature: str, algorithm: str = "sha256"
) -> bool:
    """Verify an HMAC signature.

    Args:
        key: Secret key
        message: Original message
        signature: HMAC signature to verify
        algorithm: Hash algorithm used

    Returns:
        True if signature is valid
    """
    # Calculate expected signature
    expected = create_hmac_signature(key, message, algorithm)

    # Compare signatures (constant-time comparison)
    return hmac.compare_digest(signature, expected)


def sanitize_input(text: str, allowed_patterns: Optional[List[str]] = None) -> str:
    """Sanitize user input to prevent injection attacks.

    Args:
        text: Input text to sanitize
        allowed_patterns: List of regex patterns for allowed content

    Returns:
        Sanitized input
    """
    if not text:
        return ""

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Apply allowed patterns if specified
    if allowed_patterns:
        # Filter out anything not matching allowed patterns
        filtered = ""
        for pattern in allowed_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                filtered += match.group(0)
        return filtered

    # Default sanitization (alphanumeric, spaces, and common punctuation)
    return re.sub(r'[^\w\s.,;:!?"\'-]', "", text)


def sanitize_path(path: str) -> str:
    """Sanitize file path to prevent path traversal attacks.

    Args:
        path: File path to sanitize

    Returns:
        Sanitized path
    """
    if not path:
        return ""

    # Normalize path separators
    path = path.replace("\\", "/")

    # Remove path traversal sequences
    path = re.sub(r"\.\.[/\\]", "", path)
    path = re.sub(r"[/\\]\.\.[/\\]", "/", path)

    # Remove multiple consecutive slashes
    path = re.sub(r"[/\\]{2,}", "/", path)

    # Remove leading slash
    path = re.sub(r"^[/\\]", "", path)

    # Remove dangerous characters
    path = re.sub(r'[<>:"|?*]', "", path)

    return path


def create_session_token(user_id: str, expires_in: int = 86400) -> Dict[str, Any]:
    """Create a session token for a user.

    Args:
        user_id: User identifier
        expires_in: Token expiration time in seconds

    Returns:
        Dictionary with token and expiration
    """
    # Generate expiration timestamp
    expiration = int(time.time()) + expires_in

    # Generate random token
    token = generate_random_string(48)

    # Compute signature
    # In a real implementation, use a secure key from config
    secret_key = get_env("SESSION_SECRET_KEY", "default_session_key")
    signature_msg = f"{user_id}:{token}:{expiration}"
    signature = create_hmac_signature(secret_key, signature_msg)

    return {
        "token": token,
        "signature": signature,
        "user_id": user_id,
        "expiration": expiration,
    }


def verify_session_token(token_data: Dict[str, Any]) -> bool:
    """Verify a session token.

    Args:
        token_data: Token data dictionary

    Returns:
        True if token is valid
    """
    # Check required fields
    required_fields = ["token", "signature", "user_id", "expiration"]
    if not all(field in token_data for field in required_fields):
        return False

    # Check expiration
    if int(time.time()) > token_data["expiration"]:
        return False

    # Verify signature
    secret_key = get_env("SESSION_SECRET_KEY", "default_session_key")
    signature_msg = f"{token_data['user_id']}:{token_data['token']}:{token_data['expiration']}"

    return verify_hmac_signature(secret_key, signature_msg, token_data["signature"])


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash a password securely.

    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hash, salt)
    """
    # Generate salt if not provided
    if not salt:
        salt = secrets.token_hex(16)

    # Create key derivation
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,  # 100,000 iterations
        dklen=32,
    )

    # Convert to hexadecimal
    password_hash = key.hex()

    return password_hash, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against a stored hash.

    Args:
        password: Password to verify
        stored_hash: Stored password hash
        salt: Salt used for hashing

    Returns:
        True if password is correct
    """
    # Hash the provided password with the same salt
    password_hash, _ = hash_password(password, salt)

    # Compare hashes (constant-time comparison)
    return hmac.compare_digest(password_hash, stored_hash)


def is_safe_url(url: str, allowed_hosts: Optional[List[str]] = None) -> bool:
    """Check if a URL is safe to redirect to.

    Args:
        url: URL to check
        allowed_hosts: List of allowed hosts

    Returns:
        True if URL is safe
    """
    if not url:
        return False

    # Check if URL is absolute and has a network location
    if not url.startswith(("http://", "https://")):
        # Relative URLs are considered safe
        return True

    # Parse URL
    try:
        from urllib.parse import urlparse

        parsed_url = urlparse(url)

        # Check network location
        if not parsed_url.netloc:
            return False

        # Check against allowed hosts
        if allowed_hosts:
            return parsed_url.netloc in allowed_hosts

        # Default: only allow relative URLs
        return False
    except Exception:
        return False
