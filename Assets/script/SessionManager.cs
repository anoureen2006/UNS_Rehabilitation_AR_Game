using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public enum GameState { Idle, WaitingForTarget, Tracking, Resolving }

/// <summary>
/// Top-level game loop. This is the ONLY script that should call
/// NetworkClient directly -- everyone else (BirdController,
/// GazeHeadTracker, SafeSpawnPointFinder) is purely local logic that
/// SessionManager wires together.
///
/// Attach to an empty "GameManager" GameObject alongside ARSessionManager,
/// SafeSpawnPointFinder, and GazeHeadTracker.
/// </summary>
public class SessionManager : MonoBehaviour
{
    [Header("Config")]
    [SerializeField] private string patientId = "demo01";
    [SerializeField] private string neglectSide = "left"; // "left" or "right"
    [SerializeField] private string exerciseMode = "bird_chase";

    [Header("Scene refs")]
    [SerializeField] private SafeSpawnPointFinder spawnFinder;
    [SerializeField] private GazeHeadTracker gazeTracker;
    [SerializeField] private GameObject birdPrefab;
    [SerializeField] private HUDController hud;
    [SerializeField] private StarCollectRoundController starRoundController; // only used when exerciseMode == "star_collect"

    [Header("Anchor health")]
    [Tooltip("How often to check the current anchor's tracking state, in seconds. Doesn't need to be every frame.")]
    [SerializeField] private float anchorHealthCheckIntervalS = 0.5f;

    public GameState State { get; private set; } = GameState.Idle;

    private string _sessionId;
    private GameObject _currentTarget;
    private ARAnchor _currentAnchor;
    private DifficultyParamsDto _currentDifficulty;
    private Vector3 _currentTargetLocalPosAtSpawn; // camera-local space, for logging hemifield correctly
    private Coroutine _timeoutRoutine;
    private Coroutine _anchorHealthRoutine;

    private void Start()
    {
        // Kick off the very first refresh so we have candidates ready
        // before the first target request.
        if (spawnFinder != null) spawnFinder.RefreshCandidates();

        NetworkClient.Instance.StartSession(patientId, neglectSide, exerciseMode,
            onSuccess: (resp) =>
            {
                _sessionId = resp.session_id;
                _currentDifficulty = resp.initial_difficulty;
                hud?.SetStatus($"Session {_sessionId} started");

                if (exerciseMode == "star_collect")
                {
                    // Hand off entirely to the star round controller --
                    // this script does no further per-target work in this mode.
                    starRoundController.BeginSession(_sessionId);
                }
                else
                {
                    RequestAndSpawnNextTarget(lastResult: null);
                }
            },
            onError: (err) =>
            {
                Debug.LogError(err);
                hud?.SetStatus("Failed to start session -- check backend URL/network");
            });
    }

    private void RequestAndSpawnNextTarget(TrialResultDto lastResult)
    {
        State = GameState.WaitingForTarget;
        spawnFinder.RefreshCandidates();
        var candidates = spawnFinder.GetCandidatesInCameraLocalSpace();

        NetworkClient.Instance.RequestNextTarget(_sessionId, candidates, lastResult,
            onSuccess: (resp) =>
            {
                _currentDifficulty = resp.difficulty;
                SpawnTargetAt(resp.spawn_point);
                hud?.SetDifficulty(resp.difficulty);
            },
            onError: (err) =>
            {
                Debug.LogWarning($"{err} -- retrying in 1s");
                Invoke(nameof(RetryRequestNextTarget), 1f);
            });
    }

    private void RetryRequestNextTarget() => RequestAndSpawnNextTarget(null);

    private void SpawnTargetAt(Vec3Dto spawnPointLocalSpace)
    {
        CleanupCurrentTarget();

        Camera cam = ARSessionManager.Instance.ArCamera;
        _currentTargetLocalPosAtSpawn = spawnPointLocalSpace.ToVector3();
        Vector3 worldPos = cam.transform.TransformPoint(_currentTargetLocalPosAtSpawn);

        _currentTarget = Instantiate(birdPrefab, worldPos, Quaternion.identity);

        // Anchor it so it stays fixed in real-world space regardless of the
        // patient walking around or the tracking system refining its map --
        // see the architecture notes on ARAnchor for why this matters.
        _currentAnchor = _currentTarget.AddComponent<ARAnchor>();

        var birdController = _currentTarget.GetComponent<BirdController>();
        // For bird_chase mode the target can drift slightly around its
        // spawn point to feel alive; for star_collect keep it stationary.
        Vector3 wanderTarget = worldPos + (exerciseMode == "bird_chase"
            ? Random.insideUnitSphere * 0.15f
            : Vector3.zero);
        birdController.SetTarget(wanderTarget, _currentDifficulty.speed);

        gazeTracker.BeginTrial(_currentTarget.transform);
        State = GameState.Tracking;

        gazeTracker.OnTargetAcquired += HandleTargetAcquired;

        if (_timeoutRoutine != null) StopCoroutine(_timeoutRoutine);
        _timeoutRoutine = StartCoroutine(MissTimeout(_currentDifficulty.time_limit_s));

        if (_anchorHealthRoutine != null) StopCoroutine(_anchorHealthRoutine);
        _anchorHealthRoutine = StartCoroutine(MonitorAnchorHealth());
    }

    /// <summary>
    /// Periodically checks whether the current anchor is still being
    /// actively tracked by the AR subsystem. If the patient walks
    /// somewhere with no tracked geometry, moves too fast, or the tracking
    /// system loses confidence, the anchor's TrackingState degrades from
    /// Tracking to Limited/None. When that happens we respawn the target
    /// immediately rather than let it float in a stale/wrong position --
    /// and crucially this does NOT count as a miss, since it's a tracking
    /// failure, not a patient performance failure. Logging it as a miss
    /// would corrupt your hit-rate/asymmetry metrics.
    /// </summary>
    private IEnumerator MonitorAnchorHealth()
    {
        var wait = new WaitForSeconds(anchorHealthCheckIntervalS);
        while (_currentAnchor != null)
        {
            if (_currentAnchor.trackingState != TrackingState.Tracking)
            {
                HandleAnchorLost();
                yield break;
            }
            yield return wait;
        }
    }

    private void HandleAnchorLost()
    {
        Debug.LogWarning("Anchor tracking lost -- respawning target without penalizing the trial.");
        gazeTracker.OnTargetAcquired -= HandleTargetAcquired;
        if (_timeoutRoutine != null) StopCoroutine(_timeoutRoutine);
        gazeTracker.EndTrial();
        hud?.SetStatus("Tracking interrupted -- repositioning target...");
        // Pass null (no result to log) so this doesn't affect consecutive
        // hit/miss streaks or the metrics used for the expert review.
        RequestAndSpawnNextTarget(null);
    }

    private void HandleTargetAcquired(float reactionMs, float gazeAngleDeg)
    {
        gazeTracker.OnTargetAcquired -= HandleTargetAcquired;
        if (_timeoutRoutine != null) StopCoroutine(_timeoutRoutine);
        ResolveTrial(hit: true, reactionMs, gazeAngleDeg);
    }

    private IEnumerator MissTimeout(float seconds)
    {
        yield return new WaitForSeconds(Mathf.Max(1f, seconds));
        gazeTracker.OnTargetAcquired -= HandleTargetAcquired;
        gazeTracker.EndTrial();
        ResolveTrial(hit: false, reactionMs: seconds * 1000f, gazeAngleDeg: gazeTracker.CurrentAngleDeg);
    }

    private void ResolveTrial(bool hit, float reactionMs, float gazeAngleDeg)
    {
        State = GameState.Resolving;
        gazeTracker.EndTrial();

        var result = new TrialResultDto
        {
            hit = hit,
            reaction_time_ms = reactionMs,
            gaze_angle_deg = gazeAngleDeg,
            target_position = new Vec3Dto(_currentTargetLocalPosAtSpawn)
        };

        hud?.SetLastResult(hit, reactionMs);
        RequestAndSpawnNextTarget(result);
    }

    private void CleanupCurrentTarget()
    {
        if (_anchorHealthRoutine != null) StopCoroutine(_anchorHealthRoutine);
        _anchorHealthRoutine = null;
        if (_currentTarget != null) Destroy(_currentTarget);
        _currentTarget = null;
        _currentAnchor = null;
    }

    private void OnDestroy()
    {
        if (gazeTracker != null) gazeTracker.OnTargetAcquired -= HandleTargetAcquired;
    }
}
