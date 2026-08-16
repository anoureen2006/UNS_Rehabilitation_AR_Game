using System;
using UnityEngine;

public class GazeHeadTracker : MonoBehaviour
{
    [SerializeField]
    private float acquisitionAngleThresholdDeg = 8f;

    [SerializeField]
    private float dwellTimeSeconds = 0.5f;

    public event Action<float, float>
        OnTargetAcquired;

    private Transform _target;
    private Camera _camera;

    private float _dwellTimer;
    private float _trialStartTime;

    private bool _tracking;

    public float CurrentAngleDeg
    {
        get;
        private set;
    }

    public void BeginTrial(
        Transform target
    )
    {
        if (
            ARSessionManager.Instance == null
        )
            return;

        _target = target;

        _camera =
            ARSessionManager.Instance.ArCamera;

        _dwellTimer = 0f;

        _trialStartTime =
            Time.time;

        _tracking = true;

        CurrentAngleDeg = 0f;
    }

    public void EndTrial()
    {
        _tracking = false;
        _target = null;
    }

    private void Update()
    {
        if (
            !_tracking ||
            _target == null ||
            _camera == null
        )
        {
            return;
        }

        Vector3 toTarget =
            (
                _target.position -
                _camera.transform.position
            ).normalized;

        float angle =
            Vector3.Angle(
                _camera.transform.forward,
                toTarget
            );

        CurrentAngleDeg = angle;

        if (
            angle <=
            acquisitionAngleThresholdDeg
        )
        {
            _dwellTimer +=
                Time.deltaTime;

            if (
                _dwellTimer >=
                dwellTimeSeconds
            )
            {
                float reactionMs =
                    (
                        Time.time -
                        _trialStartTime
                    ) *
                    1000f;

                _tracking = false;

                OnTargetAcquired?.Invoke(
                    reactionMs,
                    angle
                );
            }
        }
        else
        {
            _dwellTimer = 0f;
        }
    }
}