using System;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

/// <summary>
/// Central AR bootstrap. Attach this to a GameObject that also has
/// ARSession, ARPlaneManager, ARRaycastManager, and ARAnchorManager
/// components (see the Unity setup guide for exact hierarchy).
///
/// Other scripts should NOT talk to ARPlaneManager directly -- they should
/// subscribe to OnPlanesChanged here. This keeps "when do we refresh
/// candidate points" centralized in one place instead of scattered
/// per-frame polling across scripts.
/// </summary>
public class ARSessionManager : MonoBehaviour
{
    public static ARSessionManager Instance { get; private set; }

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

        if (planeManager == null) planeManager = GetComponent<ARPlaneManager>();
        if (raycastManager == null) raycastManager = GetComponent<ARRaycastManager>();
        if (anchorManager == null) anchorManager = GetComponent<ARAnchorManager>();
        if (arCamera == null) arCamera = Camera.main;
    }

    private void OnEnable()
    {
        if (planeManager != null)
            planeManager.planesChanged += HandlePlanesChanged;
    }

    private void OnDisable()
    {
        if (planeManager != null)
            planeManager.planesChanged -= HandlePlanesChanged;
    }

    private void HandlePlanesChanged(ARPlanesChangedEventArgs args)
    {
        // Only fires when planes are actually added/updated/removed --
        // NOT every frame. This is the correct place to trigger a
        // candidate-point refresh, never in an Update() loop.
        OnPlanesChanged?.Invoke();
    }
}
