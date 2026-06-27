"""Tests for logger setup."""

from tehran_house_price.utils import logger as logger_mod
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import logs_dir


def test_get_logger_returns_logger():
    log = get_logger("test_xyz")
    log.info("smoke")
    assert log.name == "test_xyz"


def test_setup_logging_creates_log_file():
    # logger should have been configured by the previous test (or first call)
    log_file = logs_dir() / "app.log"
    log = get_logger("file_check")
    log.info("trigger write")
    # flush handlers so the file actually gets touched
    for h in log.handlers + log.root.handlers:
        try:
            h.flush()
        except Exception:
            pass
    assert log_file.exists()


def test_setup_logging_is_idempotent():
    # calling twice should not crash
    logger_mod.setup_logging()
    logger_mod.setup_logging()
