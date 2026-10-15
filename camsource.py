"""
camsource.py
Shared camera layer for the latency scripts.

  --source realsense : Intel RealSense (D435) through pyrealsense2 (default)
  --source opencv    : any V4L2/UVC camera through OpenCV

Every source returns (image, age_ms):
  image  : numpy array (H,W,3 BGR for colour, H,W for infrared)
  age_ms : RealSense only - host time now minus the frame timestamp, i.e. how old
           the frame is when Python gets it (None if the timestamp is not host-synced)

    python camera_timing.py --list          # show connected RealSense devices and modes
"""
import time

import numpy as np


def add_camera_args(ap):
    g = ap.add_argument_group("camera")
    g.add_argument("--source", default="realsense", choices=["realsense", "opencv"])
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--fps", type=int, default=60)
    g.add_argument("--stream", default="color", choices=["color", "ir"],
                   help="realsense: colour (rolling shutter, max 60 fps) or left IR (global shutter, up to 90 fps at 848x480)")
    g.add_argument("--serial", default="", help="realsense: serial number if several cameras are connected")
    g.add_argument("--exposure", type=float, default=0,
                   help="realsense: manual exposure in the sensor's raw units (0 = auto); --list shows the valid range")
    g.add_argument("--cam", default="0", help="opencv: camera index or /dev/videoN")
    g.add_argument("--list", action="store_true", help="list RealSense devices and supported modes, then exit")
    return ap


def roi_mean(img, roi):
    h, w = img.shape[:2]
    dy, dx = max(1, int(h * roi / 2)), max(1, int(w * roi / 2))
    return float(img[h // 2 - dy:h // 2 + dy, w // 2 - dx:w // 2 + dx].mean())


def open_camera(a):
    if a.list:
        list_realsense()
        raise SystemExit(0)
    if a.source == "realsense":
        return RealSenseSource(a.width, a.height, a.fps, a.stream, a.serial, a.exposure)
    return OpenCVSource(a.cam, a.width, a.height, a.fps)


def _rs():
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise SystemExit("pyrealsense2 not installed:  pip install pyrealsense2")
    return rs


def list_realsense():
    rs = _rs()
    devs = rs.context().query_devices()
    if len(devs) == 0:
        print("No RealSense device found (check USB cable/port and udev rules, see README)")
        return
    for d in devs:
        info = lambda k: d.get_info(k) if d.supports(k) else "?"
        print(f"{info(rs.camera_info.name)}  serial {info(rs.camera_info.serial_number)}  "
              f"fw {info(rs.camera_info.firmware_version)}  USB {info(rs.camera_info.usb_type_descriptor)}")
        for s in d.query_sensors():
            name = s.get_info(rs.camera_info.name) if s.supports(rs.camera_info.name) else "sensor"
            modes = set()
            for p in s.get_stream_profiles():
                if p.stream_type() in (rs.stream.color, rs.stream.infrared) and p.is_video_stream_profile():
                    v = p.as_video_stream_profile()
                    modes.add((str(p.stream_type()).split(".")[-1], v.width(), v.height(), v.fps()))
            for st in sorted({m[0] for m in modes}):
                ms = sorted((m for m in modes if m[0] == st), key=lambda m: (-m[1] * m[2], -m[3]))
                print(f"  {name} / {st}: " + ", ".join(f"{w}x{h}@{f}" for _, w, h, f in ms))
            if s.supports(rs.option.exposure):
                r = s.get_option_range(rs.option.exposure)
                print(f"  {name} exposure range {r.min:.0f}..{r.max:.0f} (default {r.default:.0f})")


class RealSenseSource:
    def __init__(self, width, height, fps, stream, serial, exposure):
        rs = self.rs = _rs()
        self.ir = stream == "ir"
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        if self.ir:
            cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        else:
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipe = rs.pipeline()
        try:
            profile = self.pipe.start(cfg)
        except RuntimeError as e:
            raise SystemExit(f"RealSense: cannot start {stream} {width}x{height}@{fps}: {e}\n"
                             f"Run with --list to see supported modes (USB 2 ports allow far fewer).")
        dev = profile.get_device()
        sensor = dev.first_depth_sensor() if self.ir else dev.first_color_sensor()

        def opt(o, v):
            if sensor.supports(o):
                try:
                    sensor.set_option(o, v)
                except RuntimeError:
                    pass

        opt(rs.option.auto_exposure_priority, 0)  # never lower the fps to get a longer exposure
        if self.ir:
            opt(rs.option.emitter_enabled, 0)  # the IR dot projector would swamp the scene
        if exposure > 0:
            opt(rs.option.enable_auto_exposure, 0)
            opt(rs.option.exposure, exposure)

        usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
        self.info = (f"{dev.get_info(rs.camera_info.name)} ({stream} {width}x{height}@{fps}, USB {usb}, "
                     f"auto-exposure {'off' if exposure > 0 else 'on'})")
        if not usb.startswith("3"):
            print("WARNING: camera is not on a USB 3 link - modes and latency will be worse")
        self.domain = None

    def read(self):
        try:
            frames = self.pipe.wait_for_frames(2000)
        except RuntimeError:
            return None, None
        f = frames.get_infrared_frame(1) if self.ir else frames.get_color_frame()
        if not f:
            return None, None
        now_ms = time.time() * 1000.0
        img = np.asanyarray(f.get_data()).copy()  # copy: the SDK reuses the buffer after the frame is released
        self.domain = f.get_frame_timestamp_domain()
        host_synced = self.domain in (self.rs.timestamp_domain.global_time, self.rs.timestamp_domain.system_time)
        age = now_ms - f.get_timestamp() if host_synced else None
        return img, age

    def close(self):
        try:
            self.pipe.stop()
        except RuntimeError:
            pass


class OpenCVSource:
    def __init__(self, cam, width, height, fps):
        import cv2
        dev = int(cam) if str(cam).isdigit() else cam
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(dev)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera {cam}")
        self.info = (f"OpenCV {cam}: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
                     f"@{self.cap.get(cv2.CAP_PROP_FPS):.0f}, backend {self.cap.getBackendName()}")
        self.domain = None

    def read(self):
        ok, img = self.cap.read()
        return (img, None) if ok else (None, None)

    def close(self):
        self.cap.release()
