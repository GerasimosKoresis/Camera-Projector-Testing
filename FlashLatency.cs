// FlashLatency.cs - end-to-end latency with Unity driving the monitor/projector.
// Attach to the Camera that renders to the monitor/projector and press Play (or run a build).
// Unity clears the screen black/white; a physical camera watches it; Unity measures the time
// from "set white" until Unity's main thread knows the camera saw white.
//
//   Source = UnityWebcam  : camera read natively in Unity (WebCamTexture)
//   Source = PythonCamera : camera read by camera_publisher.py, sent to Unity over ZeroMQ
//                           together with a ~500 KB payload per frame
//
// Same display, same detection code, same clock -> the two numbers are directly comparable.
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using Debug = UnityEngine.Debug;

[RequireComponent(typeof(Camera))]
public class FlashLatency : MonoBehaviour
{
    public enum Source { UnityWebcam, PythonCamera }

    public Source source = Source.UnityWebcam;

    [Header("Display")]
    public int targetDisplay = 0;      // 1 = second display (projector); multi-display works in a build
    public bool vSync = false;         // off = minimum latency and fine-grained polling
    public int maxQueuedFrames = 1;    // fewer frames queued in the GPU = lower display latency

    [Header("Test")]
    public int trials = 50;
    [Range(0.05f, 1f)] public float roi = 0.3f;

    [Header("UnityWebcam")]
    public string webcamName = "";     // empty = first device
    public int camWidth = 640, camHeight = 480, camFps = 60;

    [Header("PythonCamera")]
    public string pythonAddress = "tcp://127.0.0.1:5557";

    Camera cam;
    WebCamTexture webcam;
    Color32[] pixels;
    readonly List<(long t, double m)> samples = new List<(long, double)>();
    readonly ConcurrentQueue<double> fromPython = new ConcurrentQueue<double>();
    Thread subThread;
    volatile bool running = true;

    void Start()
    {
        QualitySettings.vSyncCount = vSync ? 1 : 0;
        QualitySettings.maxQueuedFrames = maxQueuedFrames;
        Application.targetFrameRate = -1;

        cam = GetComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.cullingMask = 0;
        cam.backgroundColor = Color.black;
        if (targetDisplay > 0 && targetDisplay < Display.displays.Length)
        {
            Display.displays[targetDisplay].Activate();
            cam.targetDisplay = targetDisplay;
        }

        if (source == Source.UnityWebcam)
        {
            if (WebCamTexture.devices.Length == 0) { Debug.LogError("No webcam found"); return; }
            string dev = string.IsNullOrEmpty(webcamName) ? WebCamTexture.devices[0].name : webcamName;
            webcam = new WebCamTexture(dev, camWidth, camHeight, camFps);
            webcam.Play();
        }
        else
        {
            AsyncIO.ForceDotNet.Force();
            subThread = new Thread(SubLoop) { IsBackground = true };
            subThread.Start();
        }
        StartCoroutine(RunTest());
    }

    void Update()
    {
        if (source == Source.UnityWebcam)
        {
            if (webcam == null || !webcam.didUpdateThisFrame || webcam.width <= 16) return;
            int w = webcam.width, h = webcam.height;
            if (pixels == null || pixels.Length != w * h) pixels = new Color32[w * h];
            webcam.GetPixels32(pixels);
            samples.Add((Stopwatch.GetTimestamp(), RoiMean(pixels, w, h)));
        }
        else
        {
            while (fromPython.TryDequeue(out double m))
                samples.Add((Stopwatch.GetTimestamp(), m));
        }
    }

    void SubLoop()
    {
        try
        {
            using (var sub = new SubscriberSocket())
            {
                sub.Options.ReceiveHighWatermark = 4;
                sub.Connect(pythonAddress);
                sub.SubscribeToAnyTopic();
                float[] payload = null;
                while (running)
                {
                    if (!sub.TryReceiveFrameBytes(TimeSpan.FromMilliseconds(100), out byte[] msg)) continue;
                    double mean = BitConverter.ToDouble(msg, 8);    // header: int64 frame id, float64 ROI mean
                    int n = (msg.Length - 16) / 4;                  // payload: float32 vector
                    if (payload == null || payload.Length != n) payload = new float[n];
                    Buffer.BlockCopy(msg, 16, payload, 0, n * 4);   // deserialize, as a real consumer would
                    fromPython.Enqueue(mean);
                }
            }
        }
        catch (Exception e) { if (running) Debug.LogException(e); }
    }

    IEnumerator RunTest()
    {
        float deadline = Time.realtimeSinceStartup + 10f;
        while (samples.Count < 30)
        {
            if (Time.realtimeSinceStartup > deadline)
            {
                Debug.LogError("No camera samples. " + (source == Source.PythonCamera ? "Is camera_publisher.py running?" : "Check the webcam."));
                yield break;
            }
            yield return null;
        }

        SetWhite(false); yield return new WaitForSecondsRealtime(2f);   // auto-exposure settle
        double lo = LevelOverLast(0.5);
        SetWhite(true); yield return new WaitForSecondsRealtime(1f);
        double hi = LevelOverLast(0.5);
        double thr = 0.5 * (lo + hi);
        Debug.Log($"[FlashLatency {source}] black {lo:F1} white {hi:F1} threshold {thr:F1}, camera period ~{CameraPeriodMs():F1} ms");
        if (!(hi - lo > 20)) Debug.LogWarning("Low contrast: make sure the camera sees the screen in the centre of its view.");

        var lat = new List<double>();
        int frame0 = Time.frameCount;
        float time0 = Time.realtimeSinceStartup;
        for (int k = 0; k < trials; k++)
        {
            int idx = samples.Count;
            long tb = SetWhite(false);
            yield return WaitFor(idx, tb, m => m < thr, null);
            yield return new WaitForSecondsRealtime(UnityEngine.Random.Range(0.2f, 0.5f));

            idx = samples.Count;
            long t0 = SetWhite(true);
            var result = new double[] { -1 };
            yield return WaitFor(idx, t0, m => m > thr, result);
            if (result[0] < 0) Debug.LogWarning($"trial {k}: timeout");
            else lat.Add(result[0]);
        }
        double unityFps = (Time.frameCount - frame0) / (Time.realtimeSinceStartup - time0);

        if (lat.Count > 0)
            Debug.Log($"[FlashLatency {source}] Unity ran at {unityFps:F0} fps (polling resolution ~{1000 / unityFps:F1} ms)\n" +
                      Report("set white -> Unity knows", lat.ToArray()));
    }

    IEnumerator WaitFor(int idx, long t0, Func<double, bool> cond, double[] result, float timeout = 2f)
    {
        float deadline = Time.realtimeSinceStartup + timeout;
        while (Time.realtimeSinceStartup < deadline)
        {
            for (; idx < samples.Count; idx++)
            {
                var s = samples[idx];
                if (s.t > t0 && cond(s.m))
                {
                    if (result != null) result[0] = (s.t - t0) * 1000.0 / Stopwatch.Frequency;
                    yield break;
                }
            }
            yield return null;
        }
    }

    long SetWhite(bool white)
    {
        long t = Stopwatch.GetTimestamp();
        cam.backgroundColor = white ? Color.white : Color.black;
        return t;
    }

    double RoiMean(Color32[] p, int w, int h)
    {
        int dx = Math.Max(1, (int)(w * roi / 2)), dy = Math.Max(1, (int)(h * roi / 2));
        long sum = 0;
        int cnt = 0;
        for (int y = h / 2 - dy; y < h / 2 + dy; y++)
        {
            int row = y * w;
            for (int x = w / 2 - dx; x < w / 2 + dx; x++)
            {
                Color32 c = p[row + x];
                sum += c.r + c.g + c.b;
                cnt++;
            }
        }
        return cnt == 0 ? 0 : sum / (3.0 * cnt);
    }

    double LevelOverLast(double seconds)
    {
        long from = Stopwatch.GetTimestamp() - (long)(seconds * Stopwatch.Frequency);
        var v = samples.Where(s => s.t > from).Select(s => s.m).ToList();
        return v.Count > 0 ? v.Average() : double.NaN;
    }

    double CameraPeriodMs()
    {
        var d = new List<double>();
        for (int i = Math.Max(1, samples.Count - 60); i < samples.Count; i++)
            d.Add((samples[i].t - samples[i - 1].t) * 1000.0 / Stopwatch.Frequency);
        d.Sort();
        return d.Count > 0 ? d[d.Count / 2] : double.NaN;
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

    void OnDestroy()
    {
        running = false;
        if (webcam != null) webcam.Stop();
        if (subThread != null)
        {
            subThread.Join(500);
            NetMQConfig.Cleanup(false);
        }
    }
}
