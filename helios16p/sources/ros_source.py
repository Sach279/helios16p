"""RosSource: subscribe to a live ROS 2 PointCloud2 topic and yield Frames.

rclpy and sensor_msgs are imported LAZILY — inside __init__ / __iter__ — so
that `import helios16p.sources.ros_source` works in a plain Python venv that
does not have a ROS 2 installation.  rclpy is only required at the point the
user actually constructs or iterates a RosSource instance.

Usage (with ROS 2 Jazzy sourced):
    source /opt/ros/jazzy/setup.bash
    python -c "from helios16p.sources.ros_source import RosSource; ..."
"""
from __future__ import annotations

import queue
import threading
from typing import Iterator

from ..types import Frame
from .base import Source
from ._pointcloud2 import pc2_to_frame


def _require_rclpy():
    """Return the rclpy module, or raise a helpful ImportError."""
    try:
        import rclpy  # noqa: PLC0415
        return rclpy
    except ImportError as exc:
        raise ImportError(
            "rclpy is not available in the current Python environment.\n"
            "RosSource requires a sourced ROS 2 installation.  Run:\n"
            "    source /opt/ros/jazzy/setup.bash\n"
            "and then re-launch your application."
        ) from exc


class RosSource(Source):
    """Yield :class:`~helios16p.types.Frame` objects from a live ROS 2 topic.

    Subscribes to *topic* (default ``/rslidar_points``) and converts each
    incoming ``sensor_msgs/msg/PointCloud2`` message to a Frame via the shared
    :func:`~helios16p.sources._pointcloud2.pc2_to_frame` helper.

    Parameters
    ----------
    topic:
        ROS 2 topic name carrying ``sensor_msgs/msg/PointCloud2`` messages.
    node_name:
        Name given to the rclpy node that owns the subscription.
    queue_size:
        Maximum number of unprocessed messages held in the internal queue.
        Older messages are dropped when the queue is full.
    """

    name = "ros"

    def __init__(
        self,
        topic: str = "/rslidar_points",
        node_name: str = "helios16p_listener",
        queue_size: int = 10,
    ) -> None:
        # Validate that rclpy is present immediately on construction so callers
        # get a clear error at construction time rather than at first iteration.
        rclpy = _require_rclpy()

        self._topic = topic
        self._node_name = node_name
        self._queue_size = queue_size

        # State managed by __iter__ / close()
        self._msg_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._closed = False
        self._rclpy = rclpy

    # ------------------------------------------------------------------
    # Source protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        rclpy = self._rclpy

        try:
            from sensor_msgs.msg import PointCloud2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sensor_msgs is not available.\n"
                "Make sure you have sourced a ROS 2 installation:\n"
                "    source /opt/ros/jazzy/setup.bash"
            ) from exc

        # Initialise rclpy once (safe to call multiple times per process).
        if not rclpy.ok():
            rclpy.init()

        # Build node + subscription
        self._node = rclpy.create_node(self._node_name)

        msg_queue = self._msg_queue

        def _callback(msg: PointCloud2) -> None:
            try:
                msg_queue.put_nowait(msg)
            except queue.Full:
                # Drop the oldest message and enqueue the new one.
                try:
                    msg_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    msg_queue.put_nowait(msg)
                except queue.Full:
                    pass

        self._node.create_subscription(PointCloud2, self._topic, _callback, self._queue_size)

        # Spin in a background thread so __iter__ can block on the queue.
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self._node)
        self._executor = executor

        self._spin_thread = threading.Thread(target=executor.spin, daemon=True)
        self._spin_thread.start()

        frame_id = 0
        try:
            while not self._closed:
                try:
                    msg = msg_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame = pc2_to_frame(
                    width=msg.width,
                    height=msg.height,
                    point_step=msg.point_step,
                    fields=msg.fields,
                    data=bytes(msg.data),
                    is_bigendian=bool(msg.is_bigendian),
                    timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    frame_id=frame_id,
                )
                yield frame
                frame_id += 1
        finally:
            self.close()

    def close(self) -> None:
        """Stop the executor, destroy the node, and shutdown rclpy cleanly.

        Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True

        rclpy = self._rclpy

        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:  # noqa: BLE001
                pass
            self._executor = None

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            self._node = None

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
