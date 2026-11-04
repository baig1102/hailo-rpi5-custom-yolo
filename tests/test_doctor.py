import importlib
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from hailo_yolo import doctor
from hailo_yolo.doctor import (
    SetupFacts,
    check_setup,
    parse_cli_version,
    parse_dpkg,
    parse_identify,
    upstream_version,
)
from hailo_yolo.report import Finding

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


def healthy(**changes: object) -> SetupFacts:
    """The verified setup: everything at 4.20.0 on a Hailo-8L, then ``changes`` applied."""
    facts = SetupFacts(
        packages=parse_dpkg(DPKG_OK),
        cli_version="4.20.0",
        identity=parse_identify((FIXTURES / "hailo8l.txt").read_text()),
        identify_error=None,
        driver_loaded="4.20.0",
        python_version="3.11.2",
        hailo_platform_version="4.20.0",
        hailo_platform_error=None,
        numpy_version="1.26.4",
    )
    return replace(facts, **changes)  # type: ignore[arg-type]


def problems(facts: SetupFacts) -> list[Finding]:
    return [f for f in check_setup(facts) if f.level != "info"]


def test_healthy_setup_has_no_problems() -> None:
    findings = check_setup(healthy())
    assert [f for f in findings if f.level != "info"] == []
    assert any("Every Hailo part that was found is at 4.20.0" in f.message for f in findings)
    assert any("--device-arch HAILO8L" in f.message for f in findings)


def test_board_name_trap_is_explained() -> None:
    findings = check_setup(healthy())
    assert any("the chip is a Hailo-8L" in f.message for f in findings)


def test_nothing_installed() -> None:
    (first, *_) = problems(healthy(packages={}, cli_version=None))
    assert first.message == "No Hailo packages installed. Run: sudo apt install hailo-all"


def test_missing_core_package() -> None:
    packages = parse_dpkg(DPKG_OK)
    del packages["hailo-dkms"]
    (finding,) = problems(healthy(packages=packages))
    assert finding.message.startswith("Package hailo-dkms is not installed")


def test_driver_package_from_an_older_release() -> None:
    packages = {**parse_dpkg(DPKG_OK), "hailo-dkms": "4.19.0"}
    found = problems(healthy(packages=packages, driver_loaded="4.19.0"))
    assert [f.level for f in found] == ["error", "error"]
    assert "hailo-dkms is 4.19.0, HailoRT is 4.20.0" in found[0].message
    assert "Loaded PCIe driver hailo_pci is 4.19.0" in found[1].message


def test_new_driver_installed_but_not_loaded_yet() -> None:
    (finding,) = problems(healthy(driver_loaded="4.19.0"))
    assert finding.level == "error"
    assert "reboot" in finding.message


def test_firmware_from_another_minor_version_is_an_error() -> None:
    identity = parse_identify((FIXTURES / "hailo8l_fw_4_19.txt").read_text())
    (finding,) = problems(healthy(identity=identity))
    assert finding.level == "error"
    assert "Firmware running on the device is 4.19.0, HailoRT is 4.20.0" in finding.message


def test_firmware_revision_difference_is_a_warning() -> None:
    identity = replace(
        parse_identify((FIXTURES / "hailo8l.txt").read_text()), firmware_version="4.20.1"
    )
    (finding,) = problems(healthy(identity=identity))
    assert finding.level == "warning"


def test_python_bindings_from_another_version() -> None:
    (finding,) = problems(healthy(hailo_platform_version="4.19.0"))
    assert finding.level == "error"
    assert "hailo_platform in this Python is 4.19.0" in finding.message


def test_no_device() -> None:
    (finding,) = problems(
        healthy(
            identity=None,
            identify_error="[HailoRT] [error] Can't find any device\n",
            driver_loaded=None,
        )
    )
    assert finding.level == "error"
    assert "It said: [HailoRT] [error] Can't find any device" in finding.message
    assert "hailo_pci driver isn't loaded" in finding.message


def test_device_held_by_another_process() -> None:
    (finding,) = problems(
        healthy(identity=None, identify_error="status=HAILO_OUT_OF_PHYSICAL_DEVICES(74)")
    )
    assert "Another process holds the device" in finding.message


def test_hailo_platform_not_importable_in_a_plain_venv() -> None:
    (finding,) = problems(
        healthy(
            hailo_platform_version=None,
            hailo_platform_error="No module named 'hailo_platform'",
            python_version="3.12.4",
        )
    )
    assert finding.level == "error"
    assert "--system-site-packages" in finding.message
    assert "built for Python 3.11" in finding.message


def test_numpy_2_is_a_warning() -> None:
    (finding,) = problems(healthy(numpy_version="2.1.0"))
    assert finding.level == "warning"
    assert "numpy 2.1.0" in finding.message


def fake_pi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outputs: dict[str, tuple[int, str] | None],
    driver: str | None = "4.20.0\n",
    platform_version: str | None = "4.20.0",
) -> list[list[str]]:
    """Fake the commands, the driver's sysfs file and the hailo_platform import.

    ``outputs`` maps a command's first words to (exit code, stdout), or None if not installed.
    """
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        key = " ".join(cmd[:3]) if cmd[0] == "hailortcli" else cmd[0]
        result = outputs.get(key)
        if result is None:
            raise FileNotFoundError(2, "No such file or directory", cmd[0])
        return subprocess.CompletedProcess(cmd, result[0], stdout=result[1], stderr="")

    def fake_import(name: str) -> object:
        if platform_version is None:
            raise ModuleNotFoundError("No module named 'hailo_platform'")
        return types.SimpleNamespace(__version__=platform_version)

    version_file = tmp_path / "version"
    if driver is not None:
        version_file.write_text(driver)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(doctor, "DRIVER_VERSION_FILE", version_file)
    return calls


PI_OUTPUTS: dict[str, tuple[int, str] | None] = {
    "dpkg-query": (0, DPKG_OK),
    "hailortcli --version": (0, "HailoRT-CLI version 4.20.0\n"),
    "hailortcli fw-control identify": (0, (FIXTURES / "hailo8l.txt").read_text()),
}


def test_collect_facts_on_a_healthy_pi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = fake_pi(monkeypatch, tmp_path, PI_OUTPUTS)
    facts = doctor.collect_facts()
    assert facts.packages["hailo-dkms"] == "4.20.0"
    assert facts.cli_version == "4.20.0"
    assert facts.identity is not None
    assert facts.identity.device_architecture == "HAILO8L"
    assert facts.driver_loaded == "4.20.0"
    assert facts.hailo_platform_version == "4.20.0"
    assert calls[0] == ["dpkg-query", "-W", "-f", doctor.DPKG_FORMAT, *doctor.HAILO_PACKAGES]


def test_collect_facts_without_any_hailo_software(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pi(monkeypatch, tmp_path, {"dpkg-query": (1, "")}, driver=None, platform_version=None)
    facts = doctor.collect_facts()
    assert facts.packages == {}
    assert facts.cli_version is None
    assert facts.identity is None
    assert facts.identify_error == "hailortcli not found or no answer within 30 s"
    assert facts.driver_loaded is None
    assert facts.hailo_platform_error == "ModuleNotFoundError: No module named 'hailo_platform'"


def test_collect_facts_when_identify_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    outputs = {**PI_OUTPUTS, "hailortcli fw-control identify": (1, "Can't find any device\n")}
    fake_pi(monkeypatch, tmp_path, outputs)
    facts = doctor.collect_facts()
    assert facts.identity is None
    assert facts.identify_error == "Can't find any device\n"


def test_identify_that_hangs_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_pi(monkeypatch, tmp_path, PI_OUTPUTS)

    def hanging_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr(subprocess, "run", hanging_run)
    assert doctor.collect_facts().identify_error == "hailortcli not found or no answer within 30 s"


def test_main_on_a_healthy_pi_exits_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_pi(monkeypatch, tmp_path, PI_OUTPUTS)
    assert doctor.main([]) == 0
    out = capsys.readouterr().out
    assert "INFO     Every Hailo part that was found is at 4.20.0." in out


def test_main_with_problems_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_pi(monkeypatch, tmp_path, PI_OUTPUTS, driver="4.19.0\n")
    assert doctor.main([]) == 1
    assert "ERROR    Loaded PCIe driver hailo_pci is 4.19.0" in capsys.readouterr().out


def test_without_hailortcli_the_device_is_not_blamed() -> None:
    found = problems(
        healthy(
            packages={},
            cli_version=None,
            identity=None,
            identify_error="hailortcli not found or no answer within 30 s",
        )
    )
    assert "Device not checked: hailortcli isn't working." in [f.message for f in found]
    assert not any("PCIe" in f.message for f in found)


def test_device_error_detail_ends_its_own_sentence() -> None:
    (finding,) = problems(healthy(identity=None, identify_error="Can't find any device\n"))
    assert "It said: Can't find any device. Check the PCIe" in finding.message


@pytest.mark.skipif(sys.platform != "linux", reason="the script is for the Pi's bash")
def test_doctor_script_runs_from_any_directory(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "doctor.sh"
    result = subprocess.run(
        ["bash", str(script), "--help"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "usage: hailo-yolo-doctor" in result.stdout
