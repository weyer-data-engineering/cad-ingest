import os
from dataclasses import dataclass, fields

VERSION = os.environ.get("APP_VERSION", "dev")

# Previews larger than this are retried once with coarser tessellation / decimation.
PREVIEW_SOFT_LIMIT_BYTES = 50 * 1024 * 1024
# Value of the Retry-After header on every 503.
RETRY_AFTER_SECONDS = 15
# cgroup v2, then v1. A missing file, "max" or a huge value (v1) means "no limit".
CGROUP_MEMORY_FILES = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
# Share of the container memory one conversion may reserve. Not divided by the number of work-horses: RLIMIT_AS counts
# reserved address space, which is far above what a conversion really uses, so splitting it rejected ordinary files.
# The worker stays safe anyway, because work-horses set oom_score_adj=1000 and the kernel kills a horse first.
WORKER_MEMORY_SHARE = 0.85
DEFAULT_MEMORY_LIMIT_MB = 8192


def cgroup_memory_limit_bytes() -> int | None:
    for path in CGROUP_MEMORY_FILES:
        try:
            with open(path) as f:
                value = int(f.read().strip())
        except (OSError, ValueError):
            continue
        return value if value < 1 << 50 else None
    return None


@dataclass(frozen=True)
class Settings:
    """Each field is overridden by the upper-case environment variable of the same name (REDIS_URL, API_TOKEN,
    OCR_LANGUAGES, SYMBOL_LIBRARIES, DEFAULT_SYMBOL_LIBRARIES: strings, all others: integers)."""

    redis_url: str = "redis://localhost:6379/0"
    api_token: str = ""  # empty: no authentication
    max_file_size_bytes: int = 524_288_000
    conversion_timeout_seconds: int = 600
    max_concurrent_conversions: int = 2
    min_free_disk_mb: int = 1024
    max_markdown_chars: int = 2_000_000
    # Address-space cap (RLIMIT_AS) per conversion; 0 = derive from the container memory limit.
    memory_limit_mb: int = 0
    max_queued_jobs: int = 10_000
    # Runs per job, including the first one, for failures that are not the file's fault.
    max_attempts: int = 3
    result_ttl_seconds: int = 86_400
    # First delay before a re-run (then x4) and before a callback retry (then x2).
    retry_delay_seconds: int = 30
    # Tesseract languages for the texts on photos and scans, e.g. "deu+eng" (the image has deu and eng).
    ocr_languages: str = "deu"
    # Folders with symbol libraries added to the built-in ones (separated by ':'), read when a worker starts.
    symbol_libraries: str = "/libraries"
    # Installed libraries (comma-separated ids) used when a job names none; earlier ones win on equal block names.
    default_symbol_libraries: str = "din-10628"
    # Largest library a job may bring along (library_urls), per library.
    max_library_size_bytes: int = 52_428_800
    # Pages of a PDF or TIFF scan converted per job at most (each takes 5 to 60 seconds).
    max_scan_pages: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        """Reads the environment on every call, so tests can change it."""
        return cls(**{f.name: f.type(os.environ[f.name.upper()])
                      for f in fields(cls) if f.name.upper() in os.environ})

    def worker_memory_mb(self) -> int:
        """Address-space limit of one conversion: MEMORY_LIMIT_MB, else 85 % of the cgroup limit."""
        if self.memory_limit_mb:
            return self.memory_limit_mb
        limit = cgroup_memory_limit_bytes()
        if limit is None:
            return DEFAULT_MEMORY_LIMIT_MB
        return max(1, int(limit * WORKER_MEMORY_SHARE / 2**20))
