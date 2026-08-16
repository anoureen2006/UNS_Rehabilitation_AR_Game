using System.Collections;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public enum GameState
{
    Idle,
    WaitingForTarget,
    Tracking,
    Resolving
}

public class SessionManager : MonoBehaviour
{
    [Header("Config")]
    [SerializeField] private string patientId = "demo01";
    [SerializeField] private string neglectSide = "left";
    [SerializeField] private string exerciseMode = "bird_chase";

    [Header("Scene")]
    [SerializeField] private SafeSpawnPointFinder spawnFinder;
    [SerializeField] private GazeHeadTracker gazeTracker;
    [SerializeField] private GameObject birdPrefab;
    [SerializeField] private HUDController hud;
    [SerializeField] private StarCollectRoundController starRoundController;

    [Header("AR")]
    [SerializeField] private ARAnchorManager anchorManager;

    [Header("Anchor Health")]
    [SerializeField]
    private float anchorHealthCheckIntervalS = 0.5f;

    [Header("Audio Hint")]
    [SerializeField]
    private float hintDelayS = 15f;

    public GameState State { get; private set; } =
        GameState.Idle;

    private string _sessionId;

    private GameObject _currentTarget;
    private ARAnchor _currentAnchor;

    private DifficultyParamsDto _currentDifficulty;

    private Vector3 _currentTargetLocalPosAtSpawn;

    private Coroutine _timeoutRoutine;
    private Coroutine _anchorHealthRoutine;
    private Coroutine _hintRoutine;

    private int _spawnRequestId;

    private void Start()
    {
        if (anchorManager == null)
        {
            anchorManager =
                ARSessionManager.Instance?.AnchorManager;
        }

        if (anchorManager == null)
        {
            Debug.LogError(
                "SessionManager: ARAnchorManager is missing."
            );
        }

        if (PlayerPrefs.HasKey(
            "usn_neglect_side"
        ))
        {
            neglectSide =
                PlayerPrefs.GetString(
                    "usn_neglect_side",
                    neglectSide
                );

            exerciseMode =
                PlayerPrefs.GetString(
                    "usn_exercise_mode",
                    exerciseMode
                );

            patientId =
                PlayerPrefs.GetString(
                    "usn_patient_id",
                    patientId
                );
        }

        hud?.SetStatus(
            $"Patient: {patientId} | " +
            $"Neglect side: {neglectSide} | " +
            $"Mode: {exerciseMode}"
        );

        if (spawnFinder != null)
            spawnFinder.RefreshCandidates();

        if (NetworkClient.Instance == null)
        {
            Debug.LogError(
                "NetworkClient.Instance is null."
            );

            return;
        }

        NetworkClient.Instance.StartSession(
            patientId,
            neglectSide,
            exerciseMode,

            resp =>
            {
                _sessionId =
                    resp.session_id;

                _currentDifficulty =
                    resp.initial_difficulty;

                hud?.SetStatus(
                    $"Session {_sessionId} started"
                );

                if (
                    exerciseMode ==
                    "star_collect"
                )
                {
                    if (
                        starRoundController != null
                    )
                    {
                        starRoundController
                            .BeginSession(
                                _sessionId
                            );
                    }
                    else
                    {
                        Debug.LogError(
                            "StarCollectRoundController missing."
                        );
                    }
                }
                else
                {
                    RequestAndSpawnNextTarget(
                        null
                    );
                }
            },

            err =>
            {
                Debug.LogError(err);

                hud?.SetStatus(
                    "Failed to start session"
                );
            }
        );
    }

    // ============================================================
    // REQUEST TARGET
    // ============================================================

    private void RequestAndSpawnNextTarget(
        TrialResultDto lastResult
    )
    {
        if (
            string.IsNullOrEmpty(
                _sessionId
            )
        )
        {
            return;
        }

        State =
            GameState.WaitingForTarget;

        if (spawnFinder == null)
        {
            Debug.LogError(
                "SafeSpawnPointFinder missing."
            );

            return;
        }

        spawnFinder.RefreshCandidates();

        var candidates =
            spawnFinder
                .GetCandidatesInCameraLocalSpace();

        NetworkClient.Instance
            .RequestNextTarget(
                _sessionId,
                candidates,
                lastResult,

                resp =>
                {
                    _currentDifficulty =
                        resp.difficulty;

                    hud?.SetDifficulty(
                        resp.difficulty
                    );

                    _ =
                        SpawnTargetAtAsync(
                            resp.spawn_point
                        );
                },

                err =>
                {
                    Debug.LogWarning(
                        $"{err} -- retrying in 1s"
                    );

                    CancelInvoke(
                        nameof(
                            RetryRequestNextTarget
                        )
                    );

                    Invoke(
                        nameof(
                            RetryRequestNextTarget
                        ),
                        1f
                    );
                }
            );
    }

    private void RetryRequestNextTarget()
    {
        RequestAndSpawnNextTarget(
            null
        );
    }

    // ============================================================
    // SPAWN
    // ============================================================

    private async Task SpawnTargetAtAsync(
        Vec3Dto spawnPointLocalSpace
    )
    {
        if (
            spawnPointLocalSpace == null
        )
        {
            Debug.LogWarning(
                "Spawn point is null."
            );

            return;
        }

        if (anchorManager == null)
        {
            Debug.LogError(
                "ARAnchorManager missing."
            );

            return;
        }

        int requestId =
            ++_spawnRequestId;

        Camera cam =
            ARSessionManager.Instance?.ArCamera;

        if (cam == null)
        {
            Debug.LogError(
                "AR Camera missing."
            );

            return;
        }

        _currentTargetLocalPosAtSpawn =
            spawnPointLocalSpace.ToVector3();

        Vector3 worldPos =
            cam.transform.TransformPoint(
                _currentTargetLocalPosAtSpawn
            );

        // --------------------------------------------------------
        // Smooth transition
        // --------------------------------------------------------

        if (_currentTarget != null)
        {
            BirdController oldBird =
                _currentTarget
                    .GetComponent<BirdController>();

            if (oldBird != null)
            {
                await oldBird.FlyToAsync(
                    worldPos,
                    0.8f
                );
            }
        }

        if (
            requestId !=
            _spawnRequestId
        )
        {
            return;
        }

        CleanupCurrentTarget();

        await Awaitable.NextFrameAsync();

        if (
            requestId !=
            _spawnRequestId
        )
        {
            return;
        }

        Pose pose =
            new Pose(
                worldPos,
                Quaternion.identity
            );

        Result<ARAnchor> result;

        try
        {
            result =
                await anchorManager
                    .TryAddAnchorAsync(
                        pose
                    );
        }
        catch (System.Exception e)
        {
            Debug.LogWarning(
                $"Anchor creation failed: {e.Message}"
            );

            Invoke(
                nameof(
                    RetryRequestNextTarget
                ),
                1f
            );

            return;
        }

        if (
            requestId !=
            _spawnRequestId
        )
        {
            if (
                result.status.IsSuccess() &&
                result.value != null
            )
            {
                Destroy(
                    result.value.gameObject
                );
            }

            return;
        }

        if (
            !result.status.IsSuccess() ||
            result.value == null
        )
        {
            Debug.LogWarning(
                $"Anchor creation failed: {result.status}"
            );

            Invoke(
                nameof(
                    RetryRequestNextTarget
                ),
                1f
            );

            return;
        }

        _currentAnchor =
            result.value;

        // --------------------------------------------------------
        // Bird intentionally NOT parented to anchor
        // --------------------------------------------------------

        _currentTarget =
            Instantiate(
                birdPrefab,
                worldPos,
                Quaternion.identity
            );

        BirdController bird =
            _currentTarget
                .GetComponent<BirdController>();

        if (bird == null)
        {
            Debug.LogError(
                "Bird prefab does not contain BirdController."
            );

            CleanupCurrentTarget();

            return;
        }

        bird.SetTarget(
            worldPos,
            _currentDifficulty.speed
        );

        // --------------------------------------------------------
        // Gaze
        // --------------------------------------------------------

        if (gazeTracker == null)
        {
            Debug.LogError(
                "GazeHeadTracker missing."
            );

            return;
        }

        gazeTracker.BeginTrial(
            _currentTarget.transform
        );

        gazeTracker.OnTargetAcquired -=
            HandleTargetAcquired;

        gazeTracker.OnTargetAcquired +=
            HandleTargetAcquired;

        State =
            GameState.Tracking;

        // --------------------------------------------------------
        // Timeout
        // --------------------------------------------------------

        if (_timeoutRoutine != null)
        {
            StopCoroutine(
                _timeoutRoutine
            );
        }

        _timeoutRoutine =
            StartCoroutine(
                MissTimeout(
                    _currentDifficulty.time_limit_s
                )
            );

        // --------------------------------------------------------
        // Anchor health
        // --------------------------------------------------------

        if (_anchorHealthRoutine != null)
        {
            StopCoroutine(
                _anchorHealthRoutine
            );
        }

        _anchorHealthRoutine =
            StartCoroutine(
                MonitorAnchorHealth()
            );

        // --------------------------------------------------------
        // Hint
        // --------------------------------------------------------

        if (_hintRoutine != null)
        {
            StopCoroutine(
                _hintRoutine
            );
        }

        float effectiveHintDelay =
            Mathf.Min(
                hintDelayS,
                _currentDifficulty.time_limit_s *
                0.6f
            );

        if (effectiveHintDelay > 0.1f)
        {
            _hintRoutine =
                StartCoroutine(
                    HintAfterDelay(
                        effectiveHintDelay
                    )
                );
        }
    }

    // ============================================================
    // HINT
    // ============================================================

    private IEnumerator HintAfterDelay(
        float delayS
    )
    {
        yield return new WaitForSeconds(
            delayS
        );

        if (_currentTarget != null)
        {
            BirdController bird =
                _currentTarget
                    .GetComponent<BirdController>();

            bird?.PlayHintCallIfNeeded();
        }
    }

    // ============================================================
    // ANCHOR HEALTH
    // ============================================================

    private IEnumerator MonitorAnchorHealth()
    {
        const int requiredBadChecks = 4;

        int badCheckStreak = 0;

        yield return new WaitForSeconds(
            1f
        );

        WaitForSeconds wait =
            new WaitForSeconds(
                anchorHealthCheckIntervalS
            );

        while (_currentAnchor != null)
        {
            if (
                _currentAnchor.trackingState !=
                TrackingState.Tracking
            )
            {
                badCheckStreak++;

                if (
                    badCheckStreak >=
                    requiredBadChecks
                )
                {
                    HandleAnchorLost();
                    yield break;
                }
            }
            else
            {
                badCheckStreak = 0;
            }

            yield return wait;
        }
    }

    private void HandleAnchorLost()
    {
        Debug.LogWarning(
            "Anchor tracking lost -- " +
            "respawning target."
        );

        if (gazeTracker != null)
        {
            gazeTracker.OnTargetAcquired -= HandleTargetAcquired;
        }
        if (_timeoutRoutine != null)
        {
            StopCoroutine(
                _timeoutRoutine
            );
        }

        if (_hintRoutine != null)
        {
            StopCoroutine(
                _hintRoutine
            );
        }

        gazeTracker?.EndTrial();

        hud?.SetStatus(
            "Tracking interrupted..."
        );

        RequestAndSpawnNextTarget(
            null
        );
    }

    // ============================================================
    // TARGET ACQUIRED
    // ============================================================

    private void HandleTargetAcquired(
        float reactionMs,
        float gazeAngleDeg
    )
    {
        gazeTracker
            .OnTargetAcquired -=
            HandleTargetAcquired;

        if (_timeoutRoutine != null)
        {
            StopCoroutine(
                _timeoutRoutine
            );
        }

        if (_hintRoutine != null)
        {
            StopCoroutine(
                _hintRoutine
            );
        }

        ResolveTrial(
            true,
            reactionMs,
            gazeAngleDeg
        );
    }

    // ============================================================
    // MISS
    // ============================================================

    private IEnumerator MissTimeout(
        float seconds
    )
    {
        yield return new WaitForSeconds(
            Mathf.Max(
                1f,
                seconds
            )
        );

        gazeTracker
            .OnTargetAcquired -=
            HandleTargetAcquired;

        float angle =
            gazeTracker != null
                ? gazeTracker.CurrentAngleDeg
                : 0f;

        gazeTracker?.EndTrial();

        ResolveTrial(
            false,
            seconds * 1000f,
            angle
        );
    }

    // ============================================================
    // RESOLVE
    // ============================================================

    private void ResolveTrial(
        bool hit,
        float reactionMs,
        float gazeAngleDeg
    )
    {
        if (State == GameState.Resolving)
            return;

        State =
            GameState.Resolving;

        gazeTracker?.EndTrial();

        if (_currentTarget != null)
        {
            BirdController bird =
                _currentTarget
                    .GetComponent<BirdController>();

            bird?.Stop();
        }

        TrialResultDto result =
            new TrialResultDto
            {
                hit = hit,
                reaction_time_ms = reactionMs,
                gaze_angle_deg = gazeAngleDeg,
                target_position =
                    new Vec3Dto(
                        _currentTargetLocalPosAtSpawn
                    )
            };

        hud?.SetLastResult(
            hit,
            reactionMs
        );

        RequestAndSpawnNextTarget(
            result
        );
    }

    // ============================================================
    // CLEANUP
    // ============================================================

    private void CleanupCurrentTarget()
    {
        if (_timeoutRoutine != null)
        {
            StopCoroutine(
                _timeoutRoutine
            );

            _timeoutRoutine = null;
        }

        if (_anchorHealthRoutine != null)
        {
            StopCoroutine(
                _anchorHealthRoutine
            );

            _anchorHealthRoutine = null;
        }

        if (_hintRoutine != null)
        {
            StopCoroutine(
                _hintRoutine
            );

            _hintRoutine = null;
        }

        if (_currentTarget != null)
        {
            Destroy(
                _currentTarget
            );
        }

        if (_currentAnchor != null)
        {
            Destroy(
                _currentAnchor.gameObject
            );
        }

        _currentTarget = null;
        _currentAnchor = null;
    }

    private void OnDestroy()
    {
        if (gazeTracker != null)
        {
            gazeTracker
                .OnTargetAcquired -=
                HandleTargetAcquired;
        }
    }
}