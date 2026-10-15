"""
camera_publisher.py
Camera -> Python -> ZeroMQ PUB -> Unity (FlashLatency.cs with Source = PythonCamera).

Every camera frame is published as one ZeroMQ message:
    header : int64 frame id, float64 mean brightness of the centre ROI   (16 bytes, little-endian)
    payload: float32 vector of --kb kilobytes (stands in for your processed data)

Unity uses the ROI mean for flash detection, so FlashLatency measures
    Unity draws white -> display/projector -> camera -> Python -> ZeroMQ (500 KB) -> Unity

    python camera_publisher.py                     # D435 colour 640x480@60
    python camera_publisher.py --stream ir --width 848 --height 480 --fps 90
"""
import argparse
import struct
import time

import numpy as np
import zmq

from camsource import add_camera_args, open_camera, roi_mean


def main():
    ap = add_camera_args(argparse.ArgumentParser())
    ap.add_argument("--roi", type=float, default=0.3)
    ap.add_argument("--kb", type=int, default=500)
    ap.add_argument("--bind", default="tcp://*:5557")
    a = ap.parse_args()

    cam = open_camera(a)
    print(cam.info)
    payload = np.random.rand(a.kb * 1024 // 4).astype(np.float32).tobytes()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)  # drop stale frames instead of queueing them
    pub.bind(a.bind)
    print(f"publishing {len(payload) + 16} bytes per frame on {a.bind}  (Ctrl+C to stop)")

    n, t_last = 0, time.perf_counter()
    try:
        while True:
            img, _ = cam.read()
            if img is None:
                continue
            m = roi_mean(img, a.roi)
            pub.send(struct.pack("<qd", n, m) + payload)
            n += 1
            if n % 120 == 0:
                now = time.perf_counter()
                print(f"{120 / (now - t_last):5.1f} fps   roi mean {m:6.1f}")
                t_last = now
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        pub.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
