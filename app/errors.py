class ConversionError(Exception):
    """The file is unusable (corrupt, no geometry, ...). Raised by the format modules."""


class JobError(Exception):
    """Ends a job with an error code of the job API.

    `retryable` is what the client sees: submitting the job again may help. `rerun` means the cause is transient and
    the converter itself runs the job again (up to MAX_ATTEMPTS)."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, rerun: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable, self.rerun = code, message, retryable or rerun, rerun

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}
