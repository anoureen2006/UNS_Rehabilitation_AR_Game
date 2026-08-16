using System.Threading.Tasks;
using UnityEngine;

public class BirdController : MonoBehaviour
{
    private enum Motion
    {
        Idle,
        Orbiting,
        Gliding
    }

    [Header("Wander")]
    [SerializeField] private float wanderRadiusM = 0.15f;
    [SerializeField] private float verticalBobM = 0.03f;

    [Header("Orientation")]
    [SerializeField] private float forwardAxisOffsetDeg = 0f;

    [SerializeField]
    private bool feedDirectionFloatsToAnimator = false;

    [Header("Animation")]
    [SerializeField] private Animator animator;

    [Header("Audio")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip[] hintCallClips;

    private Motion _motion = Motion.Idle;

    private Vector3 _homeWorldPos;
    private float _orbitAngleRad;
    private float _phaseOffset;
    private float _speed = 0.5f;

    private Vector3 _glideStartPos;
    private Vector3 _glideDestPos;
    private float _glideDurationS;
    private float _glideElapsedS;

    private TaskCompletionSource<bool> _glideCompletion;

    private bool _hintPlayedForCurrentTarget;

    public bool HasArrived { get; private set; }

    private void Awake()
    {
        if (
            animator != null &&
            animator.applyRootMotion
        )
        {
            Debug.LogWarning(
                $"{name}: Apply Root Motion is enabled. " +
                "Disable it on the Animator."
            );
        }
    }

    public void SetTarget(
        Vector3 worldPosition,
        float speed
    )
    {
        _homeWorldPos = worldPosition;

        _speed =
            Mathf.Max(
                0.05f,
                speed
            );

        HasArrived = false;
        _hintPlayedForCurrentTarget = false;

        _orbitAngleRad = 0f;

        _phaseOffset =
            Random.Range(
                0f,
                Mathf.PI * 2f
            );

        transform.position =
            ComputeOrbitPosition(
                _orbitAngleRad
            );

        _motion = Motion.Orbiting;

        SetFlying(true);
    }

    public Task FlyToAsync(
        Vector3 destinationWorldPos,
        float durationS = 0.8f
    )
    {
        _glideStartPos = transform.position;
        _glideDestPos = destinationWorldPos;

        _glideDurationS =
            Mathf.Max(
                0.05f,
                durationS
            );

        _glideElapsedS = 0f;

        _motion = Motion.Gliding;

        SetFlying(true);

        _glideCompletion =
            new TaskCompletionSource<bool>();

        return _glideCompletion.Task;
    }

    public void Stop()
    {
        _motion = Motion.Idle;

        HasArrived = true;

        SetFlying(false);

        if (
            _glideCompletion != null &&
            !_glideCompletion.Task.IsCompleted
        )
        {
            _glideCompletion.SetResult(true);
        }
    }

    private Vector3 ComputeOrbitPosition(
        float angleRad
    )
    {
        float x =
            Mathf.Cos(
                angleRad + _phaseOffset
            ) *
            wanderRadiusM;

        float z =
            Mathf.Sin(
                angleRad + _phaseOffset
            ) *
            wanderRadiusM;

        float y =
            Mathf.Sin(
                Time.time * 1.3f +
                _phaseOffset
            ) *
            verticalBobM;

        return
            _homeWorldPos +
            new Vector3(
                x,
                y,
                z
            );
    }

    private void LateUpdate()
    {
        switch (_motion)
        {
            case Motion.Orbiting:
                {
                    float angularSpeed =
                        -(
                            _speed /
                            Mathf.Max(
                                0.02f,
                                wanderRadiusM
                            )
                        );

                    _orbitAngleRad +=
                        angularSpeed *
                        Time.deltaTime;

                    Vector3 previous =
                        transform.position;

                    Vector3 next =
                        ComputeOrbitPosition(
                            _orbitAngleRad
                        );

                    ApplyMovement(
                        previous,
                        next
                    );

                    break;
                }

            case Motion.Gliding:
                {
                    _glideElapsedS +=
                        Time.deltaTime;

                    float t =
                        Mathf.Clamp01(
                            _glideElapsedS /
                            _glideDurationS
                        );

                    Vector3 previous =
                        transform.position;

                    Vector3 next =
                        Vector3.Lerp(
                            _glideStartPos,
                            _glideDestPos,
                            t
                        );

                    ApplyMovement(
                        previous,
                        next
                    );

                    if (t >= 1f)
                    {
                        _motion = Motion.Idle;

                        if (
                            _glideCompletion != null &&
                            !_glideCompletion.Task.IsCompleted
                        )
                        {
                            _glideCompletion.SetResult(true);
                        }
                    }

                    break;
                }
        }
    }

    private void ApplyMovement(
        Vector3 previousPos,
        Vector3 nextPos
    )
    {
        transform.position = nextPos;

        Vector3 direction =
            nextPos - previousPos;

        direction.y = 0f;

        if (direction.sqrMagnitude <= 0.000001f)
            return;

        Vector3 forward =
            direction.normalized;

        Quaternion targetRotation =
            Quaternion.LookRotation(
                forward,
                Vector3.up
            );

        targetRotation *=
            Quaternion.Euler(
                0f,
                forwardAxisOffsetDeg,
                0f
            );

        transform.rotation =
            targetRotation;

        if (
            animator != null &&
            feedDirectionFloatsToAnimator
        )
        {
            Vector3 localDir =
                transform.InverseTransformDirection(
                    forward
                );

            animator.SetFloat(
                "flyingDirectionX",
                localDir.x
            );

            animator.SetFloat(
                "flyingDirectionY",
                Mathf.Abs(localDir.z)
            );
        }
    }

    public void PlayHintCallIfNeeded()
    {
        if (_hintPlayedForCurrentTarget)
            return;

        if (audioSource == null)
            return;

        if (
            hintCallClips == null ||
            hintCallClips.Length == 0
        )
            return;

        _hintPlayedForCurrentTarget = true;

        AudioClip clip =
            hintCallClips[
                Random.Range(
                    0,
                    hintCallClips.Length
                )
            ];

        audioSource.clip = clip;
        audioSource.Play();
    }

    private void SetFlying(
        bool isFlying
    )
    {
        if (animator == null)
            return;

        animator.SetBool(
            "flying",
            isFlying
        );

        animator.SetBool(
            "perched",
            !isFlying
        );
    }
}