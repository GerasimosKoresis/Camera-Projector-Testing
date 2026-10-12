"""
camera_timing.py
How long Python spends getting each frame from the camera.

  grab     : waiting for the next frame + USB/driver transfer
  retrieve : decode / convert to a numpy BGR image
  interval : time between consecutive frames (= 1/fps if Python keeps up)

This is acquisition time, not photon-to-Python latency
(use display_camera_latency.py for that).

    python camera_timing.py --cam 0 --width 640 --height 480 --fps 60 --frames 300
"""
import argparse
import time

import cv2
import numpy as np

BACKENDS = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}


def report(name, ms):
    ms = np.asarray(ms)
    print(f"{name:<26} mean {ms.mean():8.3f} | median {np.median(ms):8.3f} | std {ms.std():7.3f} | "
          f"min {ms.min():8.3f} | p95 {np.percentile(ms, 95):8.3f} | max {ms.max():8.3f}  ms (n={ms.size})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--backend", default="any", choices=list(BACKENDS))
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.cam, BACKENDS[a.backend])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ask for no frame queue (not every backend honours it)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    for _ in range(30):  # warm-up: driver start, auto-exposure
        cap.read()

    print(f"camera reports {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.1f} fps, backend {cap.getBackendName()}")

    grab, retrieve, interval = [], [], []
    frame, t_prev = None, None
    t_start = time.perf_counter()
    for _ in range(a.frames):
        t0 = time.perf_counter()
        if not cap.grab():
            continue
        t1 = time.perf_counter()
        ok, frame = cap.retrieve()
        t2 = time.perf_counter()
        if not ok:
            continue
        grab.append((t1 - t0) * 1e3)
        retrieve.append((t2 - t1) * 1e3)
        if t_prev is not None:
            interval.append((t1 - t_prev) * 1e3)
        t_prev = t1
    elapsed = time.perf_counter() - t_start
    cap.release()

    if not grab:
        raise SystemExit("No frames received")
    print(f"frame: {frame.shape}, {frame.nbytes / 1024:.0f} KB")
    report("grab (wait + transfer)", grab)
    report("retrieve (decode)", retrieve)
    report("frame interval", interval)
    print(f"effective fps: {len(grab) / elapsed:.1f}")
    print("If grab is often ~0 ms, frames were already queued (stale). If grab ~ frame period, Python is waiting on the camera.")


if __name__ == "__main__":
    main()
