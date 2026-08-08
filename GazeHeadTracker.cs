using System;
using UnityEngine;

/// <summary>
/// Computes the angle between where the patient is currently facing (AR
/// camera forward vector) and the active target each frame, purely locally
/// -- no network calls here. Fires OnTargetAcquired once the patient has
/// been looking at the target continuously for dwellTimeSeconds.
///
/// This is intentionally cheap per-frame math; only the RESULT (hit +
/// reaction time) goes over the network, via SessionManager.
/// </summary>
public class GazeHeadTracker : MonoBehaviour
{
    [SerializeField] private float acquisitionAngleThresholdDeg = 8f;
    [SerializeField] private float dwellTimeSeconds = 0.5f;

    public event Action<float, float> OnTargetAcquired; // (reactionTimeMs, gazeAngleAtAcquisitionDeg)

    private Transform _target;
    private Camera _camera;
    private float _dwellTimer = 0f;
    private float _trialStartTime = 0f;
    private bool _tracking = false;

    public float CurrentAngleDeg { get; private set; }

    public void BeginTrial(Transform target)
    {
        _target = target;
        _camera = ARSessionManager.Instance.ArCamera;
        _dwellTimer = 0f;
        _trialStartTime = Time.time;
        _tracking = true;
    }

    public void EndTrial()
    {
        _tracking = false;
        _target = null;
    }

    private void Update()
    {
        if (!_tracking || _target == null || _camera == null) return;

        Vector3 toTarget = (_target.position - _camera.transform.position).normalized;
        float angle = Vector3.Angle(_camera.transform.forward, toTarget);
        CurrentAngleDeg = angle;

        if (angle <= acquisitionAngleThresholdDeg)
        {
            _dwellTimer += Time.deltaTime;
            if (_dwellTimer >= dwellTimeSeconds)
            {
                float reactionMs = (Time.time - _trialStartTime) * 1000f;
                _tracking = false;
                OnTargetAcquired?.Invoke(reactionMs, angle);
            }
        }
        else
        {
            _dwellTimer = 0f; // must be continuous dwell, not cumulative
        }
    }
}
