"""
display_camera_latency.py
Loop latency:  Python "show white" -> monitor/projector -> light -> camera -> Python sees it.

Point the camera at the screen (or the projected image) so the image covers the centre
of the camera view. A full-screen window toggles black/white; a camera thread timestamps
every frame; latency = first bright frame time - time Python issued the white command.

Run it twice to separate the projector from the camera:
    python display_camera_latency.py --display-x 0       # monitor
    python display_camera_latency.py --display-x 1920    # projector (extended desktop, right of a 1920-px monitor)

    loop(projector) - loop(monitor)  = projector lag - monitor lag
    loop(monitor)   - monitor lag    = camera + Python acquisition latency

Press ESC in the window to abort.
"""
import argparse
import random
import threading
import time

import cv2
import numpy as np


def report(name, ms):
    ms = np.asarray(ms)
    print(f"{name:<26} mean {ms.mean():8.3f} | median {np.median(ms):8.3f} | std {ms.std():7.3f} | "
          f"min {ms.min():8.3f} | p95 {np.percentile(ms, 95):8.3f} | max {ms.max():8.3f}  ms (n={ms.size})")


class Camera(threading.Thread):
    """Reads frames as fast as the camera delivers them; stores (timestamp, mean of centre ROI)."""

    def __init__(self, index, width, height, fps, roi):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit("Cannot open camera")
        self.roi = roi
        self.samples = []  # (t, mean); list.append is atomic in CPython, so no lock needed
        self.running = True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                continue
            h, w = frame.shape[:2]
            dy, dx = max(1, int(h * self.roi / 2)), max(1, int(w * self.roi / 2))
            m = float(frame[h // 2 - dy:h // 2 + dy, w // 2 - dx:w // 2 + dx].mean())
            self.samples.append((t, m))
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--display-x", type=int, default=0, help="x offset of the target display in the virtual desktop")
    ap.add_argument("--display-y", type=int, default=0)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--roi", type=float, default=0.3, help="centre fraction of the camera frame used for brightness")
    a = ap.parse_args()

    cam = Camera(a.cam, a.width, a.height, a.fps, a.roi)
    cam.start()

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
        vals = [m for t, m in cam.samples[-300:] if t > now - seconds]
        return float(np.mean(vals)) if vals else float("nan")

    def wait_for(idx, t0, cond, timeout=2.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            n = len(cam.samples)
            for t, m in cam.samples[idx:n]:
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
        ts = [t for t, _ in cam.samples[-60:]]
        period = float(np.median(np.diff(ts))) * 1e3 if len(ts) > 2 else float("nan")
        print(f"black {lo:.1f}  white {hi:.1f}  threshold {thr:.1f}  camera frame period {period:.1f} ms")
        if not hi - lo > 20:
            print("WARNING: low contrast - make sure the camera sees the screen/projection in the centre of its view")

        lat = []
        for k in range(a.trials):
            idx = len(cam.samples)
            t = show(black)
            wait_for(idx, t, lambda m: m < thr)
            pump(random.uniform(0.2, 0.5))  # random phase w.r.t. camera and display clocks

            idx = len(cam.samples)
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
        cam.running = False
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
