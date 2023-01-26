from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    
    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)
    

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float
        
    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)
    
    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)
    
    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)
        
@dataclass
class Detection:
    rect: Rect
    class_id: int
    confidence: float
    tracker_id: Optional[int] = None
        
        
@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int
        
    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

# stores information about output video file, width and height of the frame must be equal to input video
@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int


@dataclass
class FrameData:
    pose: np.ndarray
    detection: np.ndarray


@dataclass
class Pose:
    x: np.ndarray
    y: np.ndarray
    confidence: np.ndarray

    @classmethod
    def load(cls, data: List[float]):
        x, y, confidence = [], [], []
        for i in range(17):
            x.append(data[7 + i * 3])
            y.append(data[7 + i * 3 + 1])
            confidence.append(data[7 + i * 3 + 2])
        return Pose(
            x=np.array(x),
            y=np.array(y),
            confidence=np.array(confidence)
        )


@dataclass
class Detection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Point:
        return Point(
            x=(self.x_min + self.x_max) / 2,
            y=(self.y_min + self.y_max) / 2
        )

    @classmethod
    def load(cls, data: List[float]) -> Detection:
        return Detection(
            x_min=float(data[0]),
            y_min=float(data[1]),
            x_max=float(data[2]),
            y_max=float(data[3]),
            confidence=float(data[4]),
            class_id=int(data[5])
        )

    @classmethod
    def filter(cls, detections: List[Detection], class_id: int) -> Optional[Detection]:
        filtered_detections = [
            detection
            for detection
            in detections
            if detection.class_id == class_id
        ]
        return filtered_detections[0] if len(filtered_detections) == 1 else None