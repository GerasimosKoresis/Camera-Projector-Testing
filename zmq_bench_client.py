"""
zmq_bench_client.py
ZeroMQ REQ/REP transfer benchmark, Python side, ~500 KB float32 vector.

    tcp   : Python process <-> Python process (start zmq_echo_server.py first)
            python zmq_bench_client.py --connect tcp://127.0.0.1:5555
    inproc: two threads inside one Python process (no server needed)
            python zmq_bench_client.py --connect inproc://bench

The server replies with an 8-byte ack, so
    one-way(500 KB) ~= median RTT(500 KB) - median RTT(8 B) / 2
"""
import argparse
import threading
import time

import numpy as np
import zmq

TINY = b"\x00" * 8
STOP = b"__stop__"


def report(name, ms):
    ms = np.asarray(ms)
    print(f"{name:<26} mean {ms.mean():8.3f} | median {np.median(ms):8.3f} | std {ms.std():7.3f} | "
          f"min {ms.min():8.3f} | p95 {np.percentile(ms, 95):8.3f} | max {ms.max():8.3f}  ms (n={ms.size})")


def inproc_server(ctx, addr, ready):
    rep = ctx.socket(zmq.REP)
    rep.bind(addr)
    ready.set()
    while True:
        msg = rep.recv(copy=False)
        rep.send(TINY)
        if len(msg) == len(STOP) and msg.bytes == STOP:
            break
    rep.close(linger=0)


def bench(sock, payload, n, warmup):
    for _ in range(warmup):
        sock.send(payload, copy=False)
        sock.recv()
    out = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        sock.send(payload, copy=False)
        sock.recv()
        out[i] = time.perf_counter() - t0
    return out * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp://127.0.0.1:5555")
    ap.add_argument("--kb", type=int, default=500)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    a = ap.parse_args()

    ctx = zmq.Context()
    server = None
    if a.connect.startswith("inproc://"):
        ready = threading.Event()
        server = threading.Thread(target=inproc_server, args=(ctx, a.connect, ready), daemon=True)
        server.start()
        ready.wait()

    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 2000)
    req.setsockopt(zmq.LINGER, 0)
    req.connect(a.connect)

    vec = np.random.rand(a.kb * 1024 // 4).astype(np.float32)
    print(f"target {a.connect}   payload {vec.nbytes} bytes ({vec.size} float32)")

    # Baseline: moving the same data inside one Python process, no messaging at all
    dst = np.empty_like(vec)
    cp = np.empty(a.n)
    for i in range(a.n):
        t0 = time.perf_counter()
        dst[:] = vec
        cp[i] = time.perf_counter() - t0
    report(f"memcpy {a.kb} KB (no IPC)", cp * 1e3)

    try:
        small = bench(req, TINY, a.n, a.warmup)
        big = bench(req, vec, a.n, a.warmup)
    except zmq.Again:
        raise SystemExit("No reply. Is zmq_echo_server.py running?")

    report("RTT 8 B", small)
    report(f"RTT {a.kb} KB -> ack", big)
    one_way = np.median(big) - np.median(small) / 2
    print(f"one-way {a.kb} KB (median estimate): {one_way:.3f} ms  ->  {vec.nbytes / 1e6 / (one_way / 1e3):.0f} MB/s")

    if server is not None:
        req.send(STOP)
        req.recv()
        server.join()
    req.close()
    ctx.term()


if __name__ == "__main__":
    main()
