"""Check a Raspberry Pi's Hailo setup and say in plain words what is wrong.

The parsers here are pure functions over command output, so they are tested without a Pi.
Formats, from HailoRT v4.20.0 (hailo-ai/hailort) unless noted:

- ``hailortcli --version``: "HailoRT-CLI version 4.20.0", hailort/hailortcli/hailortcli.cpp
  line 185.
- ``hailortcli fw-control identify``: hailort/hailortcli/fw_control_command.cpp lines 173-190.
- ``dpkg-query``: the format is set by ``DPKG_FORMAT`` below.
- Loaded driver version: ``/sys/module/hailo_pci/version``, set by ``MODULE_VERSION`` in
  hailo-ai/hailort-drivers v4.20.0, linux/pcie/src/pcie.c line 1562.
"""

import re
from dataclasses import dataclass

from hailo_yolo.report import Finding

# Packages that hailo-all 4.20.0+1 pulls in on Raspberry Pi OS Bookworm
# (archive.raspberrypi.com, bookworm/main, binary-arm64 Packages index).
HAILO_PACKAGES = (
    "hailo-all",
    "hailort",
    "hailo-dkms",
    "hailofw",
    "python3-hailort",
    "hailo-tappas-core",
)
# One line per package: name, version, status abbreviation ("ii " = installed).
DPKG_FORMAT = r"${Package}\t${Version}\t${db:Status-Abbrev}\n"

_VERSION = re.compile(r"(\d+\.\d+\.\d+)")


def upstream_version(debian_version: str) -> str | None:
    """The x.y.z part of a Debian version: "4.20.0-1" and "4.20.0+1" both give "4.20.0"."""
    m = _VERSION.match(debian_version.split(":", 1)[-1])  # drop an epoch like "1:"
    return m.group(1) if m else None


def parse_dpkg(text: str) -> dict[str, str]:
    """Installed packages from ``dpkg-query -W -f DPKG_FORMAT``, as name -> upstream version.

    Packages that are known but not installed (e.g. removed, config files left) are skipped.
    """
    installed = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[2].startswith("ii"):
            continue
        version = upstream_version(parts[1])
        if version is not None:
            installed[parts[0]] = version
    return installed


def parse_cli_version(text: str) -> str | None:
    """The version from ``hailortcli --version``, e.g. "4.20.0"."""
    m = re.search(r"HailoRT-CLI version (\d+\.\d+\.\d+)", text)
    return m.group(1) if m else None


@dataclass(frozen=True)
class Identity:
    firmware_version: str  # x.y.z of the firmware running on the device
    board_name: str
    device_architecture: str  # what decides which HEFs load, e.g. HAILO8L


def parse_identify(text: str) -> Identity:
    """Parse ``hailortcli fw-control identify`` output (one device)."""
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.partition(": ")
        if sep:
            fields[key.strip()] = value.strip()
    try:
        firmware = _VERSION.match(fields["Firmware Version"])
        return Identity(
            firmware_version=firmware.group(1) if firmware else fields["Firmware Version"],
            board_name=fields["Board Name"],
            device_architecture=fields["Device Architecture"],
        )
    except KeyError as e:
        raise ValueError(f"no {e} line: is this fw-control identify output?") from None


@dataclass(frozen=True)
class SetupFacts:
    """Everything the checks look at. ``None`` means the probe found nothing or failed."""

    packages: dict[str, str]  # installed Hailo packages -> x.y.z
    cli_version: str | None  # hailortcli --version
    identity: Identity | None
    identify_error: str | None  # what hailortcli printed when identify failed
    driver_loaded: str | None  # /sys/module/hailo_pci/version
    python_version: str  # the Python running the doctor, e.g. "3.11.2"
    hailo_platform_version: str | None
    hailo_platform_error: str | None
    numpy_version: str | None


# The four parts that must match HailoRT itself; hailo-tappas-core has its own numbering.
_CORE_PACKAGES = ("hailort", "hailo-dkms", "hailofw", "python3-hailort")


def check_setup(facts: SetupFacts) -> list[Finding]:
    return [
        *_check_packages(facts),
        *_check_versions(facts),
        *_check_device(facts),
        *_check_python(facts),
    ]


def _check_packages(facts: SetupFacts) -> list[Finding]:
    if not facts.packages:
        return [Finding("error", "No Hailo packages installed. Run: sudo apt install hailo-all")]
    return [
        Finding("error", f"Package {name} is not installed. Run: sudo apt install hailo-all")
        for name in _CORE_PACKAGES
        if name not in facts.packages
    ]


def _minor(version: str) -> str:
    return version.rsplit(".", 1)[0]


def _check_versions(facts: SetupFacts) -> list[Finding]:
    library = facts.cli_version or facts.packages.get("hailort")
    if library is None:
        return [Finding("error", "HailoRT not found: 'hailortcli --version' gave no version.")]

    findings = []
    # (part, its version, level if it differs, what to do). Sources: the driver must equal the
    # library exactly (hailort_driver.cpp lines 231-241, v4.20.0); the Python bindings must
    # match too (my verified setup: every part at the same version).
    parts = [
        ("hailort package", facts.packages.get("hailort"), "error", "Reinstall hailo-all."),
        ("PCIe driver package hailo-dkms", facts.packages.get("hailo-dkms"), "error",
         "Install the matching hailo-dkms."),
        ("Loaded PCIe driver hailo_pci", facts.driver_loaded, "error",
         "If hailo-dkms already matches, reboot to load it."),
        ("Firmware package hailofw", facts.packages.get("hailofw"), "error",
         "Install the matching hailofw."),
        ("python3-hailort package", facts.packages.get("python3-hailort"), "error",
         "Install the matching python3-hailort."),
        ("hailo_platform in this Python", facts.hailo_platform_version, "error",
         "This Python picks up a different hailo_platform than apt installed."),
    ]  # fmt: skip
    for part, version, level, fix in parts:
        if version is not None and version != library:
            findings.append(Finding(level, f"{part} is {version}, HailoRT is {library}. {fix}"))

    # The firmware's major.minor must match the library, or HailoRT logs "Unsupported firmware
    # operation" (device.cpp lines 456-468, v4.20.0); the revision may differ.
    if facts.identity is not None:
        running = facts.identity.firmware_version
        if _minor(running) != _minor(library):
            findings.append(
                Finding(
                    "error",
                    f"Firmware running on the device is {running}, HailoRT is {library}. "
                    "Install the matching hailofw and reboot.",
                )
            )
        elif running != library:
            findings.append(
                Finding(
                    "warning",
                    f"Firmware running on the device is {running}, HailoRT is {library}. "
                    "Reboot to load the installed firmware.",
                )
            )

    if not findings:
        # Missing parts are reported by the other checks; this only vouches for what was found.
        findings.append(Finding("info", f"Every Hailo part that was found is at {library}."))
    return findings


def _check_device(facts: SetupFacts) -> list[Finding]:
    identity = facts.identity
    if identity is None:
        hints = ["Check the PCIe connection and 'lspci | grep -i hailo'."]
        if facts.driver_loaded is None:
            hints.append("The hailo_pci driver isn't loaded (no /sys/module/hailo_pci).")
        error = facts.identify_error or ""
        if "HAILO_OUT_OF_PHYSICAL_DEVICES" in error:
            hints = ["Another process holds the device. Stop it, or run one instance only."]
        detail = f" It said: {error.strip().splitlines()[-1]}" if error.strip() else ""
        return [
            Finding(
                "error",
                f"No Hailo device answered 'hailortcli fw-control identify'.{detail} "
                + " ".join(hints),
            )
        ]

    arch = identity.device_architecture
    findings = [
        Finding(
            "info",
            f"Device: Board Name '{identity.board_name}', Device Architecture {arch}. "
            f"Check HEFs with: hailo-yolo-hefcheck model.hef --device-arch {arch}",
        )
    ]
    if arch == "HAILO8L" and identity.board_name.replace("-", "").upper() == "HAILO8":
        findings.append(
            Finding(
                "info",
                "The board name says Hailo-8, but the chip is a Hailo-8L. HEFs must be compiled "
                "for hailo8l: the Device Architecture decides, not the Board Name.",
            )
        )
    return findings


def _check_python(facts: SetupFacts) -> list[Finding]:
    findings = []
    if facts.hailo_platform_version is None:
        message = f"hailo_platform can't be imported in Python {facts.python_version}"
        if facts.hailo_platform_error:
            message += f" ({facts.hailo_platform_error})"
        message += (
            ". Use the system Python (/usr/bin/python3) or a venv created with "
            "--system-site-packages."
        )
        # python3-hailort 4.20.0-1 depends on python3 (>= 3.11~, << 3.12) in the Raspberry Pi
        # repository, i.e. it is built for Bookworm's Python 3.11.
        if not facts.python_version.startswith("3.11."):
            message += " python3-hailort is built for Python 3.11."
        findings.append(Finding("error", message))

    if facts.numpy_version is not None and int(facts.numpy_version.split(".")[0]) >= 2:
        findings.append(
            Finding(
                "warning",
                f"numpy {facts.numpy_version} in this Python. apt's hailo_platform is built "
                "against Bookworm's numpy 1.24; numpy 2 is untested with it here "
                "(the verified setup used numpy 1.26.4).",
            )
        )
    return findings
