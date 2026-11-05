"""helios16p.sources.ros_source without ROS: the module imports, and the error is clear.

With a sourced ROS 2 (rclpy importable) the constructor must succeed instead.
"""
from __future__ import annotations

import pytest

from helios16p.sources.base import Source
from helios16p.sources.ros_source import RosSource


def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401
        return True
    except ImportError:
        return False


def test_ros_source_is_a_source():
    assert issubclass(RosSource, Source)


@pytest.mark.skipif(_rclpy_available(), reason="rclpy present: the no-ROS path is not exercised")
def test_without_rclpy_the_error_names_ros():
    with pytest.raises(ImportError) as exc:
        RosSource()
    assert "ROS" in str(exc.value) or "rclpy" in str(exc.value)


@pytest.mark.skipif(not _rclpy_available(), reason="needs a sourced ROS 2")
def test_with_rclpy_construct_and_close():
    src = RosSource()
    assert isinstance(src, RosSource)
    src.close()
