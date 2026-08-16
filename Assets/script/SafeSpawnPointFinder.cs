using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class SafeSpawnPointFinder : MonoBehaviour
{
    [Header("Sampling")]
    [SerializeField] private int sampleCount = 24;

    [SerializeField]
    private float horizontalHalfAngleDeg = 90f;

    [SerializeField]
    private float verticalHalfAngleDeg = 20f;

    [Header("Distance")]
    [SerializeField] private float minDistanceM = 0.8f;
    [SerializeField] private float maxDistanceM = 3.0f;

    private readonly List<ARRaycastHit> _hitBuffer =
        new List<ARRaycastHit>();

    public List<Vector3> LatestWorldCandidates { get; private set; } =
        new List<Vector3>();

    private void OnEnable()
    {
        if (ARSessionManager.Instance != null)
        {
            ARSessionManager.Instance.OnPlanesChanged +=
                RefreshCandidates;
        }
    }

    private void OnDisable()
    {
        if (ARSessionManager.Instance != null)
        {
            ARSessionManager.Instance.OnPlanesChanged -=
                RefreshCandidates;
        }
    }

    public void RefreshCandidates()
    {
        if (ARSessionManager.Instance == null)
            return;

        ARRaycastManager raycastManager =
            ARSessionManager.Instance.RaycastManager;

        Camera cam =
            ARSessionManager.Instance.ArCamera;

        if (raycastManager == null || cam == null)
            return;

        List<Vector3> results =
            new List<Vector3>();

        Stopwatch timer =
            Stopwatch.StartNew();

        for (int i = 0; i < sampleCount; i++)
        {
            float h =
                Mathf.Lerp(
                    -horizontalHalfAngleDeg,
                    horizontalHalfAngleDeg,
                    sampleCount <= 1
                        ? 0.5f
                        : (float)i /
                          (sampleCount - 1)
                );

            float v =
                Mathf.Lerp(
                    -verticalHalfAngleDeg,
                    verticalHalfAngleDeg,
                    (i % 2 == 0)
                        ? 0.25f
                        : 0.75f
                );

            Quaternion rotation =
                Quaternion.Euler(
                    v,
                    h,
                    0f
                ) *
                cam.transform.rotation;

            Vector3 direction =
                rotation *
                Vector3.forward;

            Ray ray =
                new Ray(
                    cam.transform.position,
                    direction
                );

            _hitBuffer.Clear();

            if (
                raycastManager.Raycast(
                    ray,
                    _hitBuffer,
                    TrackableType.PlaneWithinPolygon
                )
            )
            {
                ARRaycastHit hit =
                    _hitBuffer[0];

                float distance =
                    Vector3.Distance(
                        cam.transform.position,
                        hit.pose.position
                    );

                if (
                    distance >= minDistanceM &&
                    distance <= maxDistanceM
                )
                {
                    results.Add(
                        hit.pose.position
                    );
                }
            }
        }

        timer.Stop();

        float raycastTimeMs =
            timer.ElapsedTicks *
            1000f /
            Stopwatch.Frequency;

        if (ARPerformanceMonitor.Instance != null)
        {
            ARPerformanceMonitor.Instance
                .RecordRaycastTime(
                    raycastTimeMs
                );
        }

        LatestWorldCandidates = results;
    }

    public List<Vector3> GetCandidatesInCameraLocalSpace()
    {
        if (ARSessionManager.Instance == null)
            return new List<Vector3>();

        Camera cam =
            ARSessionManager.Instance.ArCamera;

        if (cam == null)
            return new List<Vector3>();

        List<Vector3> local =
            new List<Vector3>(
                LatestWorldCandidates.Count
            );

        foreach (
            Vector3 worldPoint
            in LatestWorldCandidates
        )
        {
            local.Add(
                cam.transform.InverseTransformPoint(
                    worldPoint
                )
            );
        }

        return local;
    }
}