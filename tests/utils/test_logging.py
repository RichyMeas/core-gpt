import pytest
from pathlib import Path
from unittest.mock import mock_open, patch, call
from src.utils.logging import print0, initalize_logging

def test_print0_writes_to_file_and_console():
    mock_file = mock_open()
    test_str = "Hello, world!"

    with patch("builtins.open", mock_file), patch("builtins.print") as mock_print:
        print0(test_str, master_process=True, logfile="log.txt", console=True)

    mock_file.assert_called_once_with("log.txt", "a")
    mock_print.assert_has_calls([call(test_str), call(test_str, file=mock_file())])

def test_print0_does_not_write_when_not_master_process():
    mock_file = mock_open()
    test_str = "Hello, world!"

    with patch("builtins.open", mock_file), patch("builtins.print") as mock_print:
        print0(test_str, master_process=False, logfile="log.txt", console=True)

    mock_file.assert_not_called()
    mock_print.assert_not_called()

def test_print0_writes_to_file_only():
    mock_file = mock_open()
    test_str = "Hello, world!"

    with patch("builtins.open", mock_file), patch("builtins.print") as mock_print:
        print0(test_str, master_process=True, logfile="log.txt", console=False)

    mock_file.assert_called_once_with("log.txt", "a")
    mock_print.assert_called_once_with(test_str, file=mock_file())

def test_print0_does_not_write_when_no_logfile():
    mock_file = mock_open()
    test_str = "Hello, world!"

    with patch("builtins.open", mock_file), patch("builtins.print") as mock_print:
        print0(test_str, master_process=True, logfile=None, console=True)

    mock_file.assert_not_called()
    mock_print.assert_not_called()

def test_initalize_logging_creates_directory_and_logfile():
    mock_makedirs = patch("os.makedirs").start()
    mock_open_file = mock_open()
    patch("builtins.open", mock_open_file).start()
    patch("csv.writer").start()

    cfg = type('cfg', (object,), {})()
    cfg.model_name = "test_model"
    cfg.tokenizer = "test_tokenizer"

    # Properly mock datetime.datetime.now().strftime()
    fixed_time = "2025_01_01"
    with patch("src.utils.logging.datetime") as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = fixed_time
        initalize_logging(master_process=True, cfg=cfg)

    expected_dir = Path("experiments") / f"{fixed_time}_{cfg.model_name}"
    expected_logfile = expected_dir / "training_log.txt"

    mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
    mock_open_file.assert_any_call(expected_logfile, "a")

    # Clean up patches
    patch.stopall()