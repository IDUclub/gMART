class AuthError(Exception):
    """Base auth exception."""


class TokenExpiredError(AuthError):
    """JWT has expired."""


class InvalidTokenSignatureError(AuthError):
    """JWT signature is invalid or key not found."""


class InvalidAudienceError(AuthError):
    """JWT audience does not match the configured valid audiences."""


class AuthDecodeError(AuthError):
    """General JWT decode / processing error."""
