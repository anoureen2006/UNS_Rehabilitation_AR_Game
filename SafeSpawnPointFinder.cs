using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

/// <summary>
/// Finds candidate world-space points on detected AR planes that are within
/// a usable distance/angle range of the patient, by casting rays from
/// camera-relative viewport positions into the scene.
///
/// Candidate points are recomputed ONLY when ARSessionManager fires
/// OnPlanesChanged -- never every frame -- to keep this cheap.
/// </summary>
public class SafeSpawnPointFinder : MonoBehaviour
{
    [Header("Sampling")]
    [Tooltip("How many directions to test per refresh, spread across the FOV cone below.")]
    [SerializeField] private int sampleCount = 24;
    [Tooltip("Half-angle of the horizontal cone to search, in degrees. 90 = full left/right range.")]
    [SerializeField] private float horizontalHalfAngleDeg = 90f;
    [Tooltip("Half-angle of the vertical cone to search, in degrees.")]
    [SerializeField] private float verticalHalfAngleDeg = 20f;

    [Header("Distance filter")]
    [SerializeField] private float minDistanceM = 0.8f;
    [SerializeField] private float maxDistanceM = 3.0f;

    private readonly List<ARRaycastHit> _hitBuffer = new List<ARRaycastHit>();
    public List<Vector3> LatestWorldCandidates { get; private set; } = new List<Vector3>();

    private void OnEnable()
    {
        if (ARSessionManager.Instance != null)
            ARSessionManager.Instance.OnPlanesChanged += RefreshCandidates;
    }

    private void OnDisable()
    {
        if (ARSessionManager.Instance != null)
            ARSessionManager.Instance.OnPlanesChanged -= RefreshCandidates;
    }

    /// <summary>
    /// Re-samples the room for valid spawn points. Called on plane-changed
    /// events, and can also be called manually (e.g. right before requesting
    /// a new target) if you want a guaranteed-fresh set.
    /// </summary>
    public void RefreshCandidates()
    {
        var raycastManager = ARSessionManager.Instance.RaycastManager;
        var cam = ARSessionManager.Instance.ArCamera;
        if (raycastManager == null || cam == null) return;

        var results = new List<Vector3>();

        for (int i = 0; i < sampleCount; i++)
        {
            // Spread sample directions across the horizontal cone; a few
            // rows across the vertical cone too, so we don't only sample
            // floor-height points.
            float h = Mathf.Lerp(-horizontalHalfAngleDeg, horizontalHalfAngleDeg,
                (float)i / Mathf.Max(1, sampleCount - 1));
            float v = Mathf.Lerp(-verticalHalfAngleDeg, verticalHalfAngleDeg,
                (i % 2 == 0) ? 0.25f : 0.75f);

            Quaternion rot = Quaternion.Euler(v, h, 0f) * cam.transform.rotation;
            Vector3 direction = rot * Vector3.forward;

            // Cast from camera position outward. AR Foundation's raycast
            // manager tests against tracked plane geometry, which is what
            // makes this "safe" for Day 1 -- it can only hit real detected
            // surfaces, never empty space or through walls.
            Ray ray = new Ray(cam.transform.position, direction);
            if (raycastManager.Raycast(ray, _hitBuffer, TrackableType.PlaneWithinPolygon))
            {
                var hit = _hitBuffer[0];
                float dist = Vector3.Distance(cam.transform.position, hit.pose.position);
                if (dist >= minDistanceM && dist <= maxDistanceM)
                {
                    results.Add(hit.pose.position);
                }
            }
        }

        LatestWorldCandidates = results;
    }

    /// <summary>
    /// Converts the latest world-space candidates into the AR camera's
    /// LOCAL space, which is the coordinate convention the backend expects
    /// (see backend/schemas.py). Call this right before sending to the
    /// network layer.
    /// </summary>
    public List<Vector3> GetCandidatesInCameraLocalSpace()
    {
        var cam = ARSessionManager.Instance.ArCamera;
        var local = new List<Vector3>(LatestWorldCandidates.Count);
        foreach (var worldPoint in LatestWorldCandidates)
        {
            local.Add(cam.transform.InverseTransformPoint(worldPoint));
        }
        return local;
    }
}
