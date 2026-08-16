using System;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

/// <summary>
/// Central AR bootstrap.
/// Compatible with AR Foundation 6+.
/// </summary>
public class ARSessionManager : MonoBehaviour
{
    public static ARSessionManager Instance { get; private set; }

    [Header("AR Components")]
    [SerializeField] private ARPlaneManager planeManager;
    [SerializeField] private ARRaycastManager raycastManager;
    [SerializeField] private ARAnchorManager anchorManager;
    [SerializeField] private Camera arCamera;

    public event Action OnPlanesChanged;

    public ARPlaneManager PlaneManager => planeManager;
    public ARRaycastManager RaycastManager => raycastManager;
    public ARAnchorManager AnchorManager => anchorManager;
    public Camera ArCamera => arCamera;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;

        if (planeManager == null)
            planeManager = GetComponent<ARPlaneManager>();

        if (raycastManager == null)
            raycastManager = GetComponent<ARRaycastManager>();

        if (anchorManager == null)
            anchorManager = GetComponent<ARAnchorManager>();

        if (arCamera == null)
            arCamera = Camera.main;
    }

    private void OnEnable()
    {
        if (planeManager != null)
        {
            planeManager.trackablesChanged.AddListener(
                HandlePlanesChanged
            );
        }
    }

    private void OnDisable()
    {
        if (planeManager != null)
        {
            planeManager.trackablesChanged.RemoveListener(
                HandlePlanesChanged
            );
        }
    }

    private void HandlePlanesChanged(
        ARTrackablesChangedEventArgs<ARPlane> args
    )
    {
        OnPlanesChanged?.Invoke();
    }
}