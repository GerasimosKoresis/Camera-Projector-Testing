"""
camera_timing.py
How long Python spends getting each frame from the camera.

  read      : time blocked in read() = waiting for the next frame + transfer + conversion
  interval  : time between consecutive frames (= 1/fps if Python keeps up)
  frame age : RealSense only - host clock now minus the frame's timestamp
              (how old a frame is when Python receives it; see README for what the timestamp marks)

This is acquisition time, not photon-to-Python latency (use display_camera_latency.py for that).

    python camera_timing.py --list
    python camera_timing.py                                   # D435 colour 640x480@60
    python camera_timing.py --stream ir --width 848 --height 480 --fps 90
    python camera_timing.py --source opencv --cam /dev/video4 # D435 RGB node via V4L2
"""
import argparse
import time

import numpy as np

from camsource import add_camera_args, open_camera


def report(name, ms):
    ms = np.asarray(ms)
    print(f"{name:<26} mean {ms.mean():8.3f} | median {np.median(ms):8.3f} | std {ms.std():7.3f} | "
          f"min {ms.min():8.3f} | p95 {np.percentile(ms, 95):8.3f} | max {ms.max():8.3f}  ms (n={ms.size})")


def main():
    ap = add_camera_args(argparse.ArgumentParser())
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()

    cam = open_camera(a)
    print(cam.info)
    try:
        for _ in range(30):  # warm-up: stream start, auto-exposure
            cam.read()

        read, interval, age = [], [], []
        frame, t_prev = None, None
        t_start = time.perf_counter()
        for _ in range(a.frames):
            t0 = time.perf_counter()
            img, a_ms = cam.read()
            t1 = time.perf_counter()
            if img is None:
                continue
            frame = img
            read.append((t1 - t0) * 1e3)
            if a_ms is not None:
                age.append(a_ms)
            if t_prev is not None:
                interval.append((t1 - t_prev) * 1e3)
            t_prev = t1
        elapsed = time.perf_counter() - t_start
    finally:
        cam.close()

    if not read:
        raise SystemExit("No frames received")
    print(f"frame: {frame.shape} {frame.dtype}, {frame.nbytes / 1024:.0f} KB")
    report("read (wait + transfer)", read)
    report("frame interval", interval)
    if age:
        report("frame age at Python", age)
        print(f"timestamp domain: {cam.domain}")
    print(f"effective fps: {len(read) / elapsed:.1f}")
    print("If read is often ~0 ms, frames were already queued (stale). If read ~ frame period, Python is waiting on the camera.")


if __name__ == "__main__":
    main()