using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;

public class ARPerformanceMonitor : MonoBehaviour
{
    public static ARPerformanceMonitor Instance { get; private set; }

    [Header("Test Settings")]
    [SerializeField] private float testDuration = 30f;

    // ============================================================
    // SAMPLES
    // ============================================================

    private readonly List<float> latencySamples =
        new List<float>();

    private readonly List<float> raycastSamples =
        new List<float>();

    private readonly List<float> apiLatencySamples =
        new List<float>();

    private readonly List<float> ppoSamples =
        new List<float>();

    private readonly List<float> adaptiveDecisionSamples =
        new List<float>();

    private readonly List<float> memorySamples =
        new List<float>();

    private readonly List<float> cpuFrameSamples =
        new List<float>();

    private readonly List<float> gpuFrameSamples =
        new List<float>();

    private readonly List<float> trackingSamples =
        new List<float>();


    // ============================================================
    // TRACKING
    // ============================================================

    private Vector3 previousPosition;
    private Quaternion previousRotation;

    private bool trackingInitialized = false;


    // ============================================================
    // TEST
    // ============================================================

    private float testStartTime;

    private bool testing = false;


    // ============================================================
    // LATEST VALUES
    // ============================================================

    private float currentApiLatency;
    private float currentPpoInference;
    private float currentAdaptiveDecisionTime;


    // ============================================================
    // ACTIVE TIMERS
    // ============================================================

    private Stopwatch raycastStopwatch;
    private Stopwatch apiStopwatch;
    private Stopwatch ppoStopwatch;
    private Stopwatch adaptiveStopwatch;
    private Stopwatch endToEndStopwatch;


    // ============================================================
    // UNITY
    // ============================================================

    private void Awake()
    {
        if (
            Instance != null &&
            Instance != this
        )
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }


    private void Start()
    {
        StartTest();
    }


    private void Update()
    {
        if (!testing)
            return;

        CollectFrameMetrics();

        if (
            Time.realtimeSinceStartup -
            testStartTime >=
            testDuration
        )
        {
            FinishTest();
        }
    }


    // ============================================================
    // START TEST
    // ============================================================

    public void StartTest()
    {
        if (testing)
            return;

        ClearSamples();

        testStartTime =
            Time.realtimeSinceStartup;

        testing = true;

        UnityEngine.Debug.Log(
            "================================"
        );

        UnityEngine.Debug.Log(
            "AR PERFORMANCE TEST STARTED"
        );

        UnityEngine.Debug.Log(
            "Duration = " +
            testDuration +
            " seconds"
        );

        UnityEngine.Debug.Log(
            "================================"
        );
    }


    // ============================================================
    // FRAME METRICS
    // ============================================================

    private void CollectFrameMetrics()
    {
        // --------------------------------------------------------
        // CPU / FRAME TIME
        // --------------------------------------------------------

        float cpuFrameTime =
            Time.unscaledDeltaTime *
            1000f;

        cpuFrameSamples.Add(
            cpuFrameTime
        );


        // --------------------------------------------------------
        // MEMORY
        // --------------------------------------------------------

        float memoryMB =
            UnityEngine.Profiling.Profiler
                .GetTotalAllocatedMemoryLong()
            /
            (1024f * 1024f);

        memorySamples.Add(
            memoryMB
        );


        // --------------------------------------------------------
        // TRACKING
        // --------------------------------------------------------

        MeasureTrackingStability();
    }


    // ============================================================
    // TRACKING STABILITY
    // ============================================================

    private void MeasureTrackingStability()
    {
        Camera cam = null;


        // --------------------------------------------------------
        // TRY AR SESSION MANAGER CAMERA
        // --------------------------------------------------------

        if (
            ARSessionManager.Instance != null &&
            ARSessionManager.Instance.ArCamera != null
        )
        {
            cam =
                ARSessionManager.Instance.ArCamera;
        }


        // --------------------------------------------------------
        // FALLBACK TO MAIN CAMERA
        // --------------------------------------------------------

        if (cam == null)
        {
            cam = Camera.main;
        }


        if (cam == null)
        {
            return;
        }


        Transform cameraTransform =
            cam.transform;


        // --------------------------------------------------------
        // FIRST FRAME
        // --------------------------------------------------------

        if (!trackingInitialized)
        {
            previousPosition =
                cameraTransform.position;

            previousRotation =
                cameraTransform.rotation;

            trackingInitialized =
                true;

            return;
        }


        // --------------------------------------------------------
        // POSITION MOVEMENT
        // --------------------------------------------------------

        float positionMovement =
            Vector3.Distance(
                previousPosition,
                cameraTransform.position
            );


        // --------------------------------------------------------
        // ROTATION MOVEMENT
        // --------------------------------------------------------

        float rotationMovement =
            Quaternion.Angle(
                previousRotation,
                cameraTransform.rotation
            );


        /*
         * Tracking instability score.
         *
         * Smaller value = more stable tracking.
         *
         * Position:
         *     meters
         *
         * Rotation:
         *     degrees
         *
         * Rotation is scaled by 0.01.
         */

        float instability =
            positionMovement +
            (rotationMovement * 0.01f);


        trackingSamples.Add(
            instability
        );


        // --------------------------------------------------------
        // SAVE CURRENT CAMERA STATE
        // --------------------------------------------------------

        previousPosition =
            cameraTransform.position;

        previousRotation =
            cameraTransform.rotation;
    }


    // ============================================================
    // RAYCAST TIMER
    // ============================================================

    public void StartRaycastTimer()
    {
        if (!testing)
            return;

        raycastStopwatch =
            Stopwatch.StartNew();
    }


    public void EndRaycastTimer()
    {
        if (
            !testing ||
            raycastStopwatch == null
        )
        {
            return;
        }

        raycastStopwatch.Stop();

        float milliseconds =
            raycastStopwatch.ElapsedTicks *
            1000f /
            Stopwatch.Frequency;

        raycastSamples.Add(
            milliseconds
        );

        raycastStopwatch = null;
    }


    // ============================================================
    // DIRECT RAYCAST RECORD
    // ============================================================

    public void RecordRaycastTime(
        float milliseconds
    )
    {
        if (!testing)
            return;

        raycastSamples.Add(
            milliseconds
        );
    }


    // ============================================================
    // API TIMER
    // ============================================================

    public void StartApiTimer()
    {
        if (!testing)
            return;

        apiStopwatch =
            Stopwatch.StartNew();
    }


    public void EndApiTimer()
    {
        if (
            !testing ||
            apiStopwatch == null
        )
        {
            return;
        }

        apiStopwatch.Stop();

        float milliseconds =
            apiStopwatch.ElapsedTicks *
            1000f /
            Stopwatch.Frequency;

        RecordApiLatency(
            milliseconds
        );

        apiStopwatch = null;
    }


    // ============================================================
    // API DIRECT RECORD
    // ============================================================

    public void RecordApiLatency(
        float milliseconds
    )
    {
        if (!testing)
            return;

        currentApiLatency =
            milliseconds;

        apiLatencySamples.Add(
            milliseconds
        );
    }


    // ============================================================
    // PPO TIMER
    // ============================================================

    public void StartPPOTimer()
    {
        if (!testing)
            return;

        ppoStopwatch =
            Stopwatch.StartNew();
    }


    public void EndPPOTimer()
    {
        if (
            !testing ||
            ppoStopwatch == null
        )
        {
            return;
        }

        ppoStopwatch.Stop();

        float milliseconds =
            ppoStopwatch.ElapsedTicks *
            1000f /
            Stopwatch.Frequency;

        RecordPPOInference(
            milliseconds
        );

        ppoStopwatch = null;
    }


    // ============================================================
    // PPO DIRECT RECORD
    // ============================================================

    public void RecordPPOInference(
        float milliseconds
    )
    {
        if (!testing)
            return;

        currentPpoInference =
            milliseconds;

        ppoSamples.Add(
            milliseconds
        );
    }


    // ============================================================
    // ADAPTIVE DECISION TIMER
    // ============================================================

    public void StartAdaptiveDecisionTimer()
    {
        if (!testing)
            return;

        adaptiveStopwatch =
            Stopwatch.StartNew();
    }


    public void EndAdaptiveDecisionTimer()
    {
        if (
            !testing ||
            adaptiveStopwatch == null
        )
        {
            return;
        }

        adaptiveStopwatch.Stop();

        float milliseconds =
            adaptiveStopwatch.ElapsedTicks *
            1000f /
            Stopwatch.Frequency;

        RecordAdaptiveDecisionTime(
            milliseconds
        );

        adaptiveStopwatch = null;
    }


    // ============================================================
    // ADAPTIVE DECISION DIRECT RECORD
    // ============================================================

    public void RecordAdaptiveDecisionTime(
        float milliseconds
    )
    {
        if (!testing)
            return;

        currentAdaptiveDecisionTime =
            milliseconds;

        adaptiveDecisionSamples.Add(
            milliseconds
        );
    }


    // ============================================================
    // END-TO-END TIMER
    // ============================================================

    public void StartEndToEndTimer()
    {
        if (!testing)
            return;

        endToEndStopwatch =
            Stopwatch.StartNew();
    }


    public void EndEndToEndTimer()
    {
        if (
            !testing ||
            endToEndStopwatch == null
        )
        {
            return;
        }

        endToEndStopwatch.Stop();

        float milliseconds =
            endToEndStopwatch.ElapsedTicks *
            1000f /
            Stopwatch.Frequency;

        latencySamples.Add(
            milliseconds
        );

        endToEndStopwatch = null;
    }


    // ============================================================
    // END-TO-END DIRECT TIMER
    // ============================================================

    public Stopwatch StartEndToEndLatency()
    {
        if (!testing)
            return null;

        return Stopwatch.StartNew();
    }


    public void EndEndToEndLatency(
        Stopwatch stopwatch
    )
    {
        if (
            !testing ||
            stopwatch == null
        )
        {
            return;
        }

        stopwatch.Stop();

        float latency =
            stopwatch.ElapsedTicks *
            1000f /
            Stopwatch.Frequency;

        latencySamples.Add(
            latency
        );
    }


    // ============================================================
    // FINISH TEST
    // ============================================================

    private void FinishTest()
    {
        testing = false;


        // --------------------------------------------------------
        // STOP ACTIVE TIMERS
        // --------------------------------------------------------

        raycastStopwatch = null;
        apiStopwatch = null;
        ppoStopwatch = null;
        adaptiveStopwatch = null;
        endToEndStopwatch = null;


        UnityEngine.Debug.Log("");

        UnityEngine.Debug.Log(
            "================================"
        );

        UnityEngine.Debug.Log(
            "AR PERFORMANCE TEST FINISHED"
        );

        UnityEngine.Debug.Log(
            "================================"
        );


        PrintResults();

        SaveCSV();
    }


    // ============================================================
    // PRINT RESULTS
    // ============================================================

    private void PrintResults()
    {
        PrintMetric(
            "End-to-End Latency (ms)",
            latencySamples
        );


        PrintMetric(
            "Raycast Time (ms)",
            raycastSamples
        );


        PrintMetric(
            "API Latency (ms)",
            apiLatencySamples
        );


        PrintMetric(
            "PPO Inference Time (ms)",
            ppoSamples
        );


        PrintMetric(
            "Adaptive Decision Time (ms)",
            adaptiveDecisionSamples
        );


        PrintMetric(
            "Memory (MB)",
            memorySamples
        );


        PrintMetric(
            "CPU Frame Time (ms)",
            cpuFrameSamples
        );


        PrintMetric(
            "Tracking Instability",
            trackingSamples
        );


        // --------------------------------------------------------
        // GPU
        // --------------------------------------------------------

        UnityEngine.Debug.Log(
            "GPU Frame Time:"
        );

        UnityEngine.Debug.Log(
            "Measure GPU Frame Time using " +
            "Unity Profiler on the target device."
        );
    }


    // ============================================================
    // PRINT METRIC
    // ============================================================

    private void PrintMetric(
        string name,
        List<float> values
    )
    {
        if (
            values == null ||
            values.Count == 0
        )
        {
            UnityEngine.Debug.Log(
                name +
                " = No data"
            );

            return;
        }


        List<float> sorted =
            new List<float>(
                values
            );

        sorted.Sort();


        float average =
            Mean(sorted);


        float min =
            sorted[0];


        float max =
            sorted[
                sorted.Count - 1
            ];


        float p95 =
            Percentile(
                sorted,
                95f
            );


        float p99 =
            Percentile(
                sorted,
                99f
            );


        UnityEngine.Debug.Log(
            $"{name} | " +
            $"Average = {average:F3} | " +
            $"Min = {min:F3} | " +
            $"Max = {max:F3} | " +
            $"P95 = {p95:F3} | " +
            $"P99 = {p99:F3} | " +
            $"Samples = {sorted.Count}"
        );
    }


    // ============================================================
    // MEAN
    // ============================================================

    private float Mean(
        List<float> values
    )
    {
        if (
            values == null ||
            values.Count == 0
        )
        {
            return 0f;
        }


        float sum = 0f;


        foreach (
            float value
            in values
        )
        {
            sum += value;
        }


        return sum /
            values.Count;
    }


    // ============================================================
    // PERCENTILE
    // ============================================================

    private float Percentile(
        List<float> values,
        float percentile
    )
    {
        if (
            values == null ||
            values.Count == 0
        )
        {
            return 0f;
        }


        float index =
            (percentile / 100f) *
            (values.Count - 1);


        int lower =
            Mathf.FloorToInt(
                index
            );


        int upper =
            Mathf.CeilToInt(
                index
            );


        if (
            lower ==
            upper
        )
        {
            return values[lower];
        }


        float weight =
            index - lower;


        return
            values[lower] *
            (1f - weight)
            +
            values[upper] *
            weight;
    }


    // ============================================================
    // SAVE CSV
    // ============================================================

    private void SaveCSV()
    {
        string path =
            Path.Combine(
                Application.persistentDataPath,
                "AR_Performance_Results.csv"
            );


        using (
            StreamWriter writer =
            new StreamWriter(path)
        )
        {
            writer.WriteLine(
                "Metric,Average,Min,Max,P95,P99,Samples"
            );


            WriteCSVMetric(
                writer,
                "End-to-End Latency (ms)",
                latencySamples
            );


            WriteCSVMetric(
                writer,
                "Raycast Time (ms)",
                raycastSamples
            );


            WriteCSVMetric(
                writer,
                "API Latency (ms)",
                apiLatencySamples
            );


            WriteCSVMetric(
                writer,
                "PPO Inference Time (ms)",
                ppoSamples
            );


            WriteCSVMetric(
                writer,
                "Adaptive Decision Time (ms)",
                adaptiveDecisionSamples
            );


            WriteCSVMetric(
                writer,
                "Memory (MB)",
                memorySamples
            );


            WriteCSVMetric(
                writer,
                "CPU Frame Time (ms)",
                cpuFrameSamples
            );


            WriteCSVMetric(
                writer,
                "Tracking Instability",
                trackingSamples
            );
        }


        UnityEngine.Debug.Log(
            "================================"
        );

        UnityEngine.Debug.Log(
            "PERFORMANCE RESULTS SAVED"
        );

        UnityEngine.Debug.Log(
            path
        );

        UnityEngine.Debug.Log(
            "================================"
        );
    }


    // ============================================================
    // WRITE CSV METRIC
    // ============================================================

    private void WriteCSVMetric(
        StreamWriter writer,
        string name,
        List<float> values
    )
    {
        if (
            values == null ||
            values.Count == 0
        )
        {
            return;
        }


        List<float> sorted =
            new List<float>(
                values
            );

        sorted.Sort();


        float average =
            Mean(sorted);


        float min =
            sorted[0];


        float max =
            sorted[
                sorted.Count - 1
            ];


        float p95 =
            Percentile(
                sorted,
                95f
            );


        float p99 =
            Percentile(
                sorted,
                99f
            );


        writer.WriteLine(
            $"{name}," +
            $"{average:F3}," +
            $"{min:F3}," +
            $"{max:F3}," +
            $"{p95:F3}," +
            $"{p99:F3}," +
            $"{sorted.Count}"
        );
    }


    // ============================================================
    // CLEAR SAMPLES
    // ============================================================

    private void ClearSamples()
    {
        latencySamples.Clear();

        raycastSamples.Clear();

        apiLatencySamples.Clear();

        ppoSamples.Clear();

        adaptiveDecisionSamples.Clear();

        memorySamples.Clear();

        cpuFrameSamples.Clear();

        gpuFrameSamples.Clear();

        trackingSamples.Clear();


        trackingInitialized =
            false;


        currentApiLatency =
            0f;

        currentPpoInference =
            0f;

        currentAdaptiveDecisionTime =
            0f;


        raycastStopwatch = null;
        apiStopwatch = null;
        ppoStopwatch = null;
        adaptiveStopwatch = null;
        endToEndStopwatch = null;
    }
}