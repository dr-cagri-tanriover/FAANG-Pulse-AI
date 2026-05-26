import logging
import os
import re
from pathlib import Path

LOG_FILE = Path(__file__).parent / "runtime_logs.txt"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_RECORDS = 100
_RECORD_START = re.compile(r'(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} )')


class SizeCapRotatingHandler(logging.FileHandler):
    """Appends to runtime_logs.txt; when the file exceeds MAX_BYTES, trims to the
    last MAX_RECORDS log records and rewrites the file from scratch with those records
    as the first entries."""

    def emit(self, record):
        super().emit(record)
        self.flush()
        if os.path.getsize(self.baseFilename) >= MAX_BYTES:
            self._rotate()

    def _rotate(self):
        self.stream.close()
        content = LOG_FILE.read_text(encoding="utf-8")
        records = [r for r in _RECORD_START.split(content) if r.strip()]
        kept = records[-MAX_RECORDS:]
        LOG_FILE.write_text("".join(kept), encoding="utf-8")
        self.stream = LOG_FILE.open("a", encoding="utf-8")


_handler = SizeCapRotatingHandler(LOG_FILE, mode="a", encoding="utf-8")
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

_root = logging.getLogger("faang_pulse")
_root.setLevel(logging.INFO)
_root.addHandler(_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the faang_pulse hierarchy."""
    return logging.getLogger(f"faang_pulse.{name}")
