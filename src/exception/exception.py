class AppException(Exception):
    """Custom application exception with optional original exception chaining."""
    def __init__(self, message: str, errors: Exception = None):
        super().__init__(message)
        self.errors = errors

    def __str__(self):
        base = super().__str__()
        if self.errors:
            return f"{base} | Caused by: {repr(self.errors)}"
        return base
