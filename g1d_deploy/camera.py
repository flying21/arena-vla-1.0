"""相机采集抽象：为 VLA 提供第三人称（head）与腕部（wrist）画面。

VLA 服务端要求 ``(H, W, 3)`` 的 ``uint8`` numpy 数组（见
``run_real_eval_server.py::check_image_format``）。本模块把"从哪里拿图"抽象成
:class:`CameraSource.read`，返回规整后的 BGR→RGB 帧。

为什么不在本包内置宇树官方相机客户端
------------------------------------------------------------------
当前 ``unitree_sdk2_python`` 里 **G1 没有** 像 Go2 那样的 ``video`` 客户端子包
（G1 的头部相机通常是 Intel RealSense，经 ROS / RTSP / USB 出图）。因此：

* :class:`MockCamera` —— 合成测试图，供无相机的冒烟测试；
* :class:`WebcamCamera` —— OpenCV ``VideoCapture``，兼容 USB 相机与 RTSP 流，
  是 G1 头部相机最常见的接入方式；
* 若你的机器人已有专属相机服务，请继承 :class:`CameraSource` 实现 ``read()``，
  然后在 ``run_grasp.py`` 里换掉工厂即可。
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np


class CameraSource(abc.ABC):
    """相机源接口：``read()`` 返回一帧 ``(H, W, 3)`` ``uint8`` RGB。"""

    @abc.abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """读取一帧；无新帧/失败时返回 ``None``（由上层跳过该相机）。"""

    def close(self) -> None:
        """释放资源（默认空实现）。"""


class MockCamera(CameraSource):
    """合成相机：返回带网格的静态图，用于无硬件的链路自测。

    图像内容随时间缓慢变化（左侧色块随帧号移动），便于肉眼确认"确实在取图"。
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._counter = 0

    def read(self) -> np.ndarray:
        self._counter += 1
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # 背景网格
        img[::40, :] = 60
        img[:, ::40] = 60
        # 随时间移动的色块
        x = (self._counter * 8) % self.width
        img[:, x:x + 60] = [120, 160, 220]
        return img


class WebcamCamera(CameraSource):
    """OpenCV 相机：支持 ``VideoCapture(索引)`` 或 RTSP/HTTP 流地址。

    Args:
        source: 相机索引（整数）或流地址（字符串）。
        width/height: 期望分辨率（0 表示不设置）。
    """

    def __init__(self, source=0, width: int = 640, height: int = 480) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._cap = None

    def _ensure(self):
        if self._cap is None:
            import cv2

            self._cap = cv2.VideoCapture(self.source)
            if self.width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self._cap

    def read(self) -> Optional[np.ndarray]:
        import cv2

        cap = self._ensure()
        ok, frame = cap.read()
        if not ok:
            return None
        # OpenCV 默认 BGR → 转成 VLA 期望的 RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def build_camera(kind: str, **kwargs) -> CameraSource:
    """相机工厂。

    Args:
        kind: ``"mock"`` / ``"webcam"``。
        kwargs: 透传给对应相机类的构造参数。

    Returns:
        :class:`CameraSource` 实例。
    """
    if kind == "mock":
        return MockCamera(**kwargs)
    if kind == "webcam":
        return WebcamCamera(**kwargs)
    raise ValueError(f"未知相机类型: {kind}（可选 'mock' / 'webcam'）")
