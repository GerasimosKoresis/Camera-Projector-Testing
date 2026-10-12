"""
camera_publisher.py
Camera -> Python -> ZeroMQ PUB -> Unity (FlashLatency.cs with Source = PythonCamera).

Every camera frame is published as one ZeroMQ message:
    header : int64 frame id, float64 mean brightness of the centre ROI   (16 bytes, little-endian)
    payload: float32 vector of --kb kilobytes (stands in for your processed data)

Unity uses the ROI mean for flash detection, so FlashLatency measures
    Unity draws white -> display/projector -> camera -> Python -> ZeroMQ (500 KB) -> Unity

    python camera_publisher.py --cam 0 --bind tcp://*:5557 --kb 500
"""
import argparse
import struct
import time

import cv2
import numpy as np
import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--roi", type=float, default=0.3)
    ap.add_argument("--kb", type=int, default=500)
    ap.add_argument("--bind", default="tcp://*:5557")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    payload = np.random.rand(a.kb * 1024 // 4).astype(np.float32).tobytes()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)  # drop stale frames instead of queueing them
    pub.bind(a.bind)
    print(f"publishing {len(payload) + 16} bytes per frame on {a.bind}  (Ctrl+C to stop)")

    n, t_last = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            dy, dx = max(1, int(h * a.roi / 2)), max(1, int(w * a.roi / 2))
            m = float(frame[h // 2 - dy:h // 2 + dy, w // 2 - dx:w // 2 + dx].mean())
            pub.send(struct.pack("<qd", n, m) + payload)
            n += 1
            if n % 120 == 0:
                now = time.perf_counter()
                print(f"{120 / (now - t_last):5.1f} fps   roi mean {m:6.1f}")
                t_last = now
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pub.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
