"""
display_camera_latency.py
Loop latency:  Python "show white" -> monitor/projector -> light -> camera -> Python sees it.

Point the camera at the screen (or the projected image) so the image covers the centre
of the camera view. A full-screen window toggles black/white; a camera thread timestamps
every frame; latency = first bright frame time - time Python issued the white command.

Run it twice to separate the projector from the camera (x offsets from `xrandr --listmonitors`):
    python display_camera_latency.py --display-x 0       # monitor
    python display_camera_latency.py --display-x 1920    # projector to the right of a 1920-px monitor

    loop(projector) - loop(monitor)  = projector lag - monitor lag
    loop(monitor)   - monitor lag    = camera + Python acquisition latency

Use an X11 session (Ubuntu on Xorg): under Wayland the window cannot be moved to another display.
Press ESC in the window to abort.
"""
import argparse
import random
import threading
import time

import cv2
import numpy as np

from camsource import add_camera_args, open_camera, roi_mean


def report(name, ms):
    ms = np.asarray(ms)
    print(f"{name:<26} mean {ms.mean():8.3f} | median {np.median(ms):8.3f} | std {ms.std():7.3f} | "
          f"min {ms.min():8.3f} | p95 {np.percentile(ms, 95):8.3f} | max {ms.max():8.3f}  ms (n={ms.size})")


class CameraThread(threading.Thread):
    """Reads frames as fast as the camera delivers them; stores (timestamp, mean of centre ROI)."""

    def __init__(self, cam, roi):
        super().__init__(daemon=True)
        self.cam, self.roi = cam, roi
        self.samples = []  # (t, mean); list.append is atomic in CPython, so no lock needed
        self.running = True

    def run(self):
        while self.running:
            img, _ = self.cam.read()
            t = time.perf_counter()
            if img is not None:
                self.samples.append((t, roi_mean(img, self.roi)))


def main():
    ap = add_camera_args(argparse.ArgumentParser())
    ap.add_argument("--display-x", type=int, default=0, help="x offset of the target display in the X11 desktop")
    ap.add_argument("--display-y", type=int, default=0)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--roi", type=float, default=0.3, help="centre fraction of the camera frame used for brightness")
    a = ap.parse_args()

    cam = open_camera(a)
    print(cam.info)
    ct = CameraThread(cam, a.roi)
    ct.start()

    win = "latency"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, a.display_x, a.display_y)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black = np.zeros((16, 16, 3), np.uint8)  # tiny image, scaled to full screen by the window
    white = np.full((16, 16, 3), 255, np.uint8)

    def show(img):
        t = time.perf_counter()  # "Python issued the command"
        cv2.imshow(win, img)
        cv2.waitKey(1)  # HighGUI paints here
        return t

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            if cv2.waitKey(1) == 27:
                raise KeyboardInterrupt

    def level(seconds):
        now = time.perf_counter()
        vals = [m for t, m in ct.samples[-300:] if t > now - seconds]
        return float(np.mean(vals)) if vals else float("nan")

    def wait_for(idx, t0, cond, timeout=2.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            n = len(ct.samples)
            for t, m in ct.samples[idx:n]:
                if t > t0 and cond(m):
                    return t
            idx = n
            if cv2.waitKey(1) == 27:
                raise KeyboardInterrupt
        return None

    try:
        show(black)
        pump(2.0)  # let the camera start and auto-exposure settle
        lo = level(0.5)
        show(white)
        pump(1.0)
        hi = level(0.5)
        thr = 0.5 * (lo + hi)
        ts = [t for t, _ in ct.samples[-60:]]
        period = float(np.median(np.diff(ts))) * 1e3 if len(ts) > 2 else float("nan")
        print(f"black {lo:.1f}  white {hi:.1f}  threshold {thr:.1f}  camera frame period {period:.1f} ms")
        if not hi - lo > 20:
            print("WARNING: low contrast - make sure the camera sees the screen/projection in the centre of its view")

        lat = []
        for k in range(a.trials):
            idx = len(ct.samples)
            t = show(black)
            wait_for(idx, t, lambda m: m < thr)
            pump(random.uniform(0.2, 0.5))  # random phase w.r.t. camera and display clocks

            idx = len(ct.samples)
            t0 = show(white)
            t1 = wait_for(idx, t0, lambda m: m > thr)
            if t1 is None:
                print(f"trial {k}: timeout")
                continue
            lat.append((t1 - t0) * 1e3)
            print(f"trial {k:3d}: {lat[-1]:7.2f} ms")

        if lat:
            report("command -> camera sees it", lat)
            print(f"(includes up to one camera frame period of quantisation, ~{period:.1f} ms)")
    except KeyboardInterrupt:
        print("aborted")
    finally:
        ct.running = False
        ct.join(timeout=3)
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
