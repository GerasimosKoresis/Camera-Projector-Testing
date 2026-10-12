// ZmqBench.cs - ZeroMQ (NetMQ) transfer benchmark, Unity side, ~500 KB float32 vector.
// Attach to any GameObject and press Play; results appear in the Console.
//
//   Target = PythonTcp   : Unity <-> Python process   (run zmq_echo_server.py first)
//   Target = UnityInproc : two threads inside Unity, inproc://   (no Python)
//   Target = UnityTcp    : two threads inside Unity, tcp loopback (no Python)
//
// The server replies with an 8-byte ack, so
//   one-way(500 KB) ~= median RTT(500 KB) - median RTT(8 B) / 2
using System;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class ZmqBench : MonoBehaviour
{
    public enum Target { PythonTcp, UnityInproc, UnityTcp }

    public Target target = Target.PythonTcp;
    public string pythonAddress = "tcp://127.0.0.1:5555";
    public int payloadKB = 500;
    public int iterations = 1000;
    public int warmup = 50;

    static readonly byte[] Tiny = new byte[8];
    static readonly byte[] Stop = Encoding.ASCII.GetBytes("__stop__");

    void Start()
    {
        AsyncIO.ForceDotNet.Force();                       // required for NetMQ inside Unity
        new Thread(Run) { IsBackground = true }.Start();   // keep blocking sockets off the main thread
    }

    void OnDestroy()
    {
        NetMQConfig.Cleanup(false);                        // otherwise the Editor can hang on exit
    }

    void Run()
    {
        string addr = target == Target.PythonTcp ? pythonAddress
                    : target == Target.UnityInproc ? "inproc://bench"
                    : "tcp://127.0.0.1:5556";
        try
        {
            if (target != Target.PythonTcp)
            {
                var ready = new ManualResetEventSlim(false);
                new Thread(() => EchoServer(addr, ready)) { IsBackground = true }.Start();
                ready.Wait();
            }

            int n = payloadKB * 1024 / 4;
            var vec = new float[n];
            var rng = new System.Random(0);
            for (int i = 0; i < n; i++) vec[i] = (float)rng.NextDouble();
            var bytes = new byte[n * 4];

            // Baseline: float[] -> byte[] inside Unity (a 500 KB memcpy, no messaging)
            var ser = new double[iterations];
            for (int i = 0; i < iterations; i++)
            {
                long t0 = Stopwatch.GetTimestamp();
                Buffer.BlockCopy(vec, 0, bytes, 0, bytes.Length);
                ser[i] = Ms(t0);
            }

            using (var req = new RequestSocket())
            {
                req.Connect(addr);
                double[] small = Bench(req, Tiny);
                double[] big = Bench(req, bytes);
                if (target != Target.PythonTcp)
                {
                    req.SendFrame(Stop);
                    req.TryReceiveFrameBytes(TimeSpan.FromSeconds(1), out _);
                }

                double oneWay = Median(big) - Median(small) / 2;
                Debug.Log($"[ZmqBench {target}] target {addr}, payload {bytes.Length} bytes\n" +
                          Report($"BlockCopy {payloadKB} KB (no IPC)", ser) + "\n" +
                          Report("RTT 8 B", small) + "\n" +
                          Report($"RTT {payloadKB} KB -> ack", big) + "\n" +
                          $"one-way {payloadKB} KB (median estimate): {oneWay:F3} ms -> {bytes.Length / 1e6 / (oneWay / 1e3):F0} MB/s");
            }
        }
        catch (Exception e) { Debug.LogException(e); }
    }

    double[] Bench(RequestSocket req, byte[] payload)
    {
        for (int i = 0; i < warmup; i++) RoundTrip(req, payload);
        var ms = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            RoundTrip(req, payload);
            ms[i] = Ms(t0);
        }
        return ms;
    }

    static void RoundTrip(RequestSocket req, byte[] payload)
    {
        req.SendFrame(payload);
        if (!req.TryReceiveFrameBytes(TimeSpan.FromSeconds(2), out _))
            throw new TimeoutException("No reply. Is zmq_echo_server.py running?");
    }

    static void EchoServer(string addr, ManualResetEventSlim ready)
    {
        using (var rep = new ResponseSocket())
        {
            rep.Bind(addr);
            ready.Set();
            while (true)
            {
                byte[] msg = rep.ReceiveFrameBytes();
                rep.SendFrame(Tiny);
                if (msg.Length == Stop.Length && msg.SequenceEqual(Stop)) break;
            }
        }
    }

    static double Ms(long t0) => (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;

    static double Median(double[] x)
    {
        var s = (double[])x.Clone();
        Array.Sort(s);
        return s[s.Length / 2];
    }

    static string Report(string name, double[] x)
    {
        var s = (double[])x.Clone();
        Array.Sort(s);
        double mean = s.Average();
        double std = Math.Sqrt(s.Select(v => (v - mean) * (v - mean)).Average());
        double p95 = s[(int)(0.95 * (s.Length - 1))];
        return $"{name,-28} mean {mean,8:F3} | median {s[s.Length / 2],8:F3} | std {std,7:F3} | " +
               $"min {s[0],8:F3} | p95 {p95,8:F3} | max {s[s.Length - 1],8:F3} ms (n={s.Length})";
    }
}
