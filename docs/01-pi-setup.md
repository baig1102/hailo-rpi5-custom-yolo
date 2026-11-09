# 1. Set up the Pi

From a fresh Raspberry Pi OS to a Hailo device that answers, plus this repo's checks.

**What this is based on.** My verified setup: Raspberry Pi OS **Bookworm** 64-bit with
HailoRT **4.20** (see the table in the [README](../README.md)). Raspberry Pi's
[AI software page](https://www.raspberrypi.com/documentation/computers/ai.html) now
targets **Trixie**, whose repository ships HailoRT 4.23 and builds `python3-hailort` for
Python 3.13. The steps below are the same on both, but this repo is only tested against
the 4.20 behaviour.

## Hardware

Fit the AI Kit or AI HAT+ as described on Raspberry Pi's
[AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html) or
[AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) page.
Use a 64-bit OS on a Raspberry Pi 5.

## Install

These commands come from Raspberry Pi's AI software page ("Software prerequisites").

```bash
sudo apt update
sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo reboot
```

```bash
sudo apt install dkms
sudo apt install hailo-all
sudo reboot
```

`hailo-all` installs HailoRT and `hailortcli`, the PCIe driver, the firmware, the Python
bindings (`hailo_platform`) and TAPPAS core. The reboot loads the driver and the firmware.

### PCIe Gen 3 (AI Kit only)

The Pi 5's PCIe runs at Gen 2 by default. For the AI Kit, Raspberry Pi recommends Gen 3
(on the AI HAT+ it is applied automatically). Either use `sudo raspi-config` →
*Advanced Options* → *PCIe Speed*, or add this line to `/boot/firmware/config.txt` and
reboot:

```
dtparam=pciex1_gen=3
```

My setup used this line. If you later see `HAILO_DRIVER_OPERATION_FAILED` again and
again, try without it (Gen 2); see [troubleshooting](troubleshooting.md).

## Verify

```bash
hailortcli fw-control identify
```

Expected, roughly (the numbers depend on your release):

```
Executing on device: 0000:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.20.0 (release,app,extended context switch buffer)
Logger Version: 0
Board Name: Hailo-8
Device Architecture: HAILO8L
Serial Number: <N/A>
Part Number: <N/A>
Product Name: <N/A>
```

Two things to read here:

- **`Device Architecture`, not `Board Name`, says which chip you have.** A Hailo-8L
  reports `Board Name: Hailo-8`. Compile your model for the *architecture*.
- **`<N/A>`** for the serial, part and product numbers is normal on the AI HAT+
  (Raspberry Pi's page says so).

The kernel log should show the driver loading and the device being added:

```bash
dmesg | grep -i hailo
```

## Get this repo and run the doctor

```bash
git clone https://github.com/baig1102/hailo-rpi5-custom-yolo.git
cd hailo-rpi5-custom-yolo
./scripts/doctor.sh
```

The doctor uses the system Python, so it sees the `hailo_platform` that `apt`
installed, and it needs no `pip install`. It checks the following:

- **Versions:** HailoRT, the PCIe driver (installed *and* loaded), the firmware
  (installed *and* running) and the Python bindings all have the same version.
- **Device:** a device answers, and its architecture is shown.
- **Python:** `hailo_platform` imports in this Python.

Fix every `ERROR` before going on. Each line says what to do.

## Install the package (for the `hailo-yolo-*` commands and the examples)

Raspberry Pi OS doesn't allow `pip install` into the system Python, so use a virtual
environment. Create it **with `--system-site-packages`**, so that the `hailo_platform`,
numpy, OpenCV and Picamera2 from `apt` stay visible inside it:

```bash
sudo apt install python3-opencv python3-picamera2
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
.venv/bin/hailo-yolo-doctor
```

A venv created without that flag can't import `hailo_platform`. The doctor says so.

Don't install `numpy` 2 or OpenCV with `pip` into this venv: `apt`'s `hailo_platform`
is built against the system numpy (1.24 on Bookworm).

## Keep the versions together

Every Hailo part must stay at the same version (the doctor checks this). After an
`apt upgrade`, reboot and run `./scripts/doctor.sh` again. To hold a working set,
Raspberry Pi's page shows `apt-mark hold` on the Hailo packages.

Next: [run your model](02-run-your-model.md).
