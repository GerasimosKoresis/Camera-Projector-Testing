"""
zmq_echo_server.py
ZeroMQ REP server used by zmq_bench_client.py (Python client) and ZmqBench.cs (Unity client).
Replies to every request with an 8-byte ack, so the client measures:
    RTT(8 B)     = pure messaging overhead
    RTT(500 KB)  = moving 500 KB one way + a tiny ack back

    python zmq_echo_server.py --bind tcp://*:5555
"""
import argparse

import zmq

ACK = b"\x00" * 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://*:5555")
    a = ap.parse_args()

    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.bind(a.bind)
    print(f"echo server on {a.bind}  (Ctrl+C to stop)")
    try:
        while True:
            rep.recv(copy=False)  # zero-copy receive: we only measure transport, not Python copies
            rep.send(ACK)
    except KeyboardInterrupt:
        pass
    finally:
        rep.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
