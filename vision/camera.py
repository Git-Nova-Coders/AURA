"""
AURA Camera Adapter Module
Provides an abstraction layer for frame capture from webcams, video files, or test streams.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Structured representation of a video/camera frame."""
    image: np.ndarray
    timestamp: float
    source_id: str

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape


class CameraError(Exception):
    """Base exception for camera-related errors."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when the requested camera device or video source cannot be found/opened."""
    pass


class CameraAdapter:
    """
    Abstracted camera capture interface.
    
    Supports:
    - Webcams (via device index, e.g., 0, 1)
    - Video files or network streams (via string path / URL)
    - Context management protocol (`with CameraAdapter(...) as cam:`)
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        source_id: Optional[str] = None,
    ):
        self.source = source
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.source_id = source_id or (f"webcam_{source}" if isinstance(source, int) else str(source))
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "CameraAdapter":
        """
        Opens the video capture device.
        
        Raises:
            CameraNotFoundError: If the camera device cannot be opened.
        """
        if self.is_opened:
            return self

        logger.info(f"Opening camera source: {self.source} ({self.source_id})")
        
        # On Windows, using cv2.CAP_DSHOW or default CAP_ANY for webcams
        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                # Fallback to default backend
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(str(self.source))

        if not self._cap.isOpened():
            self._cap = None
            raise CameraNotFoundError(
                f"Failed to open video source '{self.source}'. "
                "Verify camera is connected, permissions are granted, or file path exists."
            )

        # Set capture properties if specified
        if self.requested_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        logger.info(
            f"Camera opened successfully: {self.width}x{self.height} @ {self.fps:.1f} FPS"
        )
        return self

    def read(self) -> Optional[Frame]:
        """
        Captures and returns the next frame from the stream.
        
        Returns:
            Frame: The captured frame with metadata.
            None: If the stream has ended or reading failed.
        """
        if not self.is_opened or self._cap is None:
            raise CameraError("Camera is not opened. Call open() first.")

        ret, image = self._cap.read()
        if not ret or image is None or image.size == 0:
            return None

        return Frame(
            image=image,
            timestamp=time.time(),
            source_id=self.source_id,
        )

    def release(self) -> None:
        """Releases the camera device and frees system resources."""
        if self._cap is not None:
            logger.info(f"Releasing camera source: {self.source_id}")
            self._cap.release()
            self._cap = None

    @property
    def is_opened(self) -> bool:
        """Returns True if the camera device is currently opened."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        """Returns the current frame width."""
        if self._cap is not None and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 0

    @property
    def height(self) -> int:
        """Returns the current frame height."""
        if self._cap is not None and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0

    @property
    def fps(self) -> float:
        """Returns the current frame rate reported by the capture device."""
        if self._cap is not None and self._cap.isOpened():
            fps_val = self._cap.get(cv2.CAP_PROP_FPS)
            return fps_val if fps_val > 0 else 30.0
        return 0.0

    def __enter__(self) -> "CameraAdapter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __repr__(self) -> str:
        status = "opened" if self.is_opened else "closed"
        return f"<CameraAdapter(source='{self.source}', status={status}, resolution={self.width}x{self.height})>"
