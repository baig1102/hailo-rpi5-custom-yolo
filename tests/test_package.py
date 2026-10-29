from importlib.metadata import version

import hailo_yolo


def test_version_matches_installed_metadata() -> None:
    assert hailo_yolo.__version__ == version("hailo-yolo")
