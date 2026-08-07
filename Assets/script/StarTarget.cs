using System;
using UnityEngine;

/// <summary>
/// Attach to each spawned star prefab in star_collect mode. Unlike
/// GazeHeadTracker (built for one bird at a time), several of these run
/// simultaneously during a round, each independently checking whether the
/// patient is currently looking at IT specifically.
///
/// StarCollectRoundController owns the round-level state (how many
/// collected, when the round ends) -- this script only knows about itself.
/// </summary>
public class StarTarget : MonoBehaviour
{
    [SerializeField] private float acquisitionAngleThresholdDeg = 8f;
    [SerializeField] private float dwellTimeSeconds = 0.4f; // slightly shorter than bird mode since multiple targets compete for attention
    [SerializeField] private GameObject collectedVisualEffect; // optional, leave null if you don't have a particle/flash prefab yet

    public event Action<StarTarget, float, float> OnCollected; // (self, reactionTimeMs, gazeAngleAtCollectionDeg)

    public bool IsCollected { get; private set; } = false;
    public Vector3 SpawnLocalPos { get; private set; } // camera-local position at spawn time, for backend logging

    private float _dwellTimer = 0f;
    private float _spawnTime;
    private Camera _camera;

    public void Initialize(Vector3 spawnLocalPos)
    {
        SpawnLocalPos = spawnLocalPos;
        _spawnTime = Time.time;
        _camera = ARSessionManager.Instance.ArCamera;
        IsCollected = false;
        _dwellTimer = 0f;
    }

    private void Update()
    {
        if (IsCollected || _camera == null) return;

        Vector3 toTarget = (transform.position - _camera.transform.position).normalized;
        float angle = Vector3.Angle(_camera.transform.forward, toTarget);

        if (angle <= acquisitionAngleThresholdDeg)
        {
            _dwellTimer += Time.deltaTime;
            if (_dwellTimer >= dwellTimeSeconds)
            {
                Collect(angle);
            }
        }
        else
        {
            _dwellTimer = 0f;
        }
    }

    private void Collect(float gazeAngleDeg)
    {
        IsCollected = true;
        float reactionMs = (Time.time - _spawnTime) * 1000f;

        if (collectedVisualEffect != null)
            Instantiate(collectedVisualEffect, transform.position, Quaternion.identity);

        gameObject.SetActive(false); // hide immediately, destroyed later by the round controller
        OnCollected?.Invoke(this, reactionMs, gazeAngleDeg);
    }

    /// <summary>Called by the round controller if time runs out before this star was collected.</summary>
    public float TimeSinceSpawnMs() => (Time.time - _spawnTime) * 1000f;
}
