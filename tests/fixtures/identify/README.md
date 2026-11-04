# Synthetic `hailortcli fw-control identify` outputs

Written by hand for the tests, **not** captured from a device. They follow the text printed by
HailoRT v4.20.0 (hailo-ai/hailort): `hailort/hailortcli/command.cpp` line 64 ("Executing on
device") and `hailort/hailortcli/fw_control_command.cpp` lines 173-190. Serial, part and
product numbers are `<N/A>`, what HailoRT prints when they aren't set (same file, line 15).

| File | What it shows |
|---|---|
| `hailo8l.txt` | The board name trap: `Board Name: Hailo-8` on a `HAILO8L` |
| `hailo8l_fw_4_19.txt` | Same device, firmware 4.19.0 still running |
