from pathlib import Path

import pytest

from hailo_yolo.doctor import (
    parse_cli_version,
    parse_dpkg,
    parse_identify,
    upstream_version,
)

FIXTURES = Path(__file__).parent / "fixtures" / "identify"

# Output of dpkg-query with DPKG_FORMAT on a Pi with hailo-all 4.20 (synthetic).
DPKG_OK = (
    "hailo-all\t4.20.0+1\tii \n"
    "hailort\t4.20.0-1\tii \n"
    "hailo-dkms\t4.20.0-1\tii \n"
    "hailofw\t4.20.0-1\tii \n"
    "python3-hailort\t4.20.0-1\tii \n"
    "hailo-tappas-core\t3.31.0+1-1\tii \n"
)


@pytest.mark.parametrize(
    ("debian", "upstream"),
    [
        ("4.20.0-1", "4.20.0"),
        ("4.20.0+1", "4.20.0"),
        ("3.31.0+1-1", "3.31.0"),
        ("1:4.6.0", "4.6.0"),
    ],
)
def test_upstream_version(debian: str, upstream: str) -> None:
    assert upstream_version(debian) == upstream


def test_upstream_version_without_x_y_z() -> None:
    assert upstream_version("garbage") is None


def test_parse_dpkg_installed_packages() -> None:
    assert parse_dpkg(DPKG_OK) == {
        "hailo-all": "4.20.0",
        "hailort": "4.20.0",
        "hailo-dkms": "4.20.0",
        "hailofw": "4.20.0",
        "python3-hailort": "4.20.0",
        "hailo-tappas-core": "3.31.0",
    }


def test_parse_dpkg_skips_removed_packages() -> None:
    # "rc": removed, config files still there.
    text = "hailort\t4.20.0-1\tii \nhailo-dkms\t4.19.0-1\trc \n"
    assert parse_dpkg(text) == {"hailort": "4.20.0"}


def test_parse_cli_version() -> None:
    assert parse_cli_version("HailoRT-CLI version 4.20.0\n") == "4.20.0"
    assert parse_cli_version("bash: hailortcli: command not found") is None


def test_parse_identify() -> None:
    identity = parse_identify((FIXTURES / "hailo8l.txt").read_text())
    assert identity.firmware_version == "4.20.0"
    assert identity.board_name == "Hailo-8"
    assert identity.device_architecture == "HAILO8L"


def test_parse_identify_rejects_other_text() -> None:
    with pytest.raises(ValueError, match="is this fw-control identify output"):
        parse_identify("[HailoRT] [error] Can't find any device")
