import logging
import os
import re
import sys
import traceback
from pathlib import Path

LOG_FILE = Path(__file__).parent / "faang_pulse_ai_runtime_logs.txt"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_RECORDS = 100
_UPLOAD_EVERY = 10
_RECORD_START = re.compile(r'(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} )')

_HF_DATASET_REPO = "ML-Owl/app-runtime-logs"
_HF_LOG_PATH     = "faang_pulse_ai_runtime_logs.txt"
_ON_HF = bool(os.environ.get("SPACE_ID")) and bool(os.environ.get("TOKEN_HF_PULSE_AI"))


def _restore_from_hf():
    """Download the prior log file from the HF dataset into LOG_FILE before the
    file handler opens it. Silently skipped on any error (missing file, network
    failure, etc.) so a fresh empty log is used instead."""
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        cached = hf_hub_download(
            repo_id=_HF_DATASET_REPO,
            filename=_HF_LOG_PATH,
            repo_type="dataset",
            token=os.environ.get("TOKEN_HF_PULSE_AI"),
            force_download=True,
        )
        shutil.copy(cached, LOG_FILE)
    except Exception:
        pass


def _upload_to_hf():
    """Upload the current log file to the HF dataset in a daemon thread so it
    never blocks the logging call that triggered rotation."""
    import threading

    def _do():
        try:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=str(LOG_FILE),
                path_in_repo=_HF_LOG_PATH,
                repo_id=_HF_DATASET_REPO,
                repo_type="dataset",
                token=os.environ.get("TOKEN_HF_PULSE_AI"),
                commit_message="log rotation",
            )
        except Exception:
            pass

    threading.Thread(target=_do, daemon=True).start()


class SizeCapRotatingHandler(logging.FileHandler):
    """Appends to the log file; when the file exceeds MAX_BYTES, trims to the
    last MAX_RECORDS log records and rewrites the file from scratch with those
    records as the first entries. On HF Spaces, also uploads after each rotation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._emit_count = 0

    def handleError(self, record):
        print(
            "SizeCapRotatingHandler: emit() failed — file logging is broken:",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)

    def emit(self, record):
        super().emit(record)
        self.flush()
        if _ON_HF:
            self._emit_count += 1
            if self._emit_count >= _UPLOAD_EVERY:
                _upload_to_hf()
                self._emit_count = 0
        if os.path.getsize(self.baseFilename) >= MAX_BYTES:
            self._rotate()

    def _rotate(self):
        self.stream.close()
        content = LOG_FILE.read_text(encoding="utf-8")
        records = [r for r in _RECORD_START.split(content) if r.strip()]
        kept = records[-MAX_RECORDS:]
        LOG_FILE.write_text("".join(kept), encoding="utf-8")
        self.stream = LOG_FILE.open("a", encoding="utf-8")
        if _ON_HF:
            _upload_to_hf()


if _ON_HF:
    _restore_from_hf()

_handler = SizeCapRotatingHandler(LOG_FILE, mode="a", encoding="utf-8")
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

_root = logging.getLogger("faang_pulse")
_root.setLevel(logging.INFO)
_root.addHandler(_handler)
_root.addHandler(_stream_handler)
_root.info("faang_pulse logger initialized — log file: %s", LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the faang_pulse hierarchy."""
    return logging.getLogger(f"faang_pulse.{name}")
