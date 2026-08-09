using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
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

    [Header("AR refs")]
    [Tooltip("Drag the ARAnchorManager component (usually lives on the XR Origin GameObject) here. Required -- anchors must be created through the manager, never via AddComponent<ARAnchor>().")]
    [SerializeField] private ARAnchorManager anchorManager;

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

    // Guards against a stale/in-flight spawn finishing after a newer one
    // has already started (e.g. anchor-lost fires right as a trial resolves).
    private int _spawnRequestId;

    private void Start()
    {
        if (anchorManager == null)
        {
            Debug.LogError("SessionManager: anchorManager is not assigned. " +
                            "Drag the ARAnchorManager (on your XR Origin) into the AR refs field.");
        }

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
                hud?.SetDifficulty(resp.difficulty);
                _ = SpawnTargetAtAsync(resp.spawn_point); // fire-and-forget; guarded internally by _spawnRequestId
            },
            onError: (err) =>
            {
                Debug.LogWarning($"{err} -- retrying in 1s");
                Invoke(nameof(RetryRequestNextTarget), 1f);
            });
    }

    private void RetryRequestNextTarget() => RequestAndSpawnNextTarget(null);

    /// <summary>
    /// Destroys any existing target/anchor, waits for that destruction to
    /// actually flush (Destroy() is deferred to end-of-frame -- creating a
    /// new anchor before the old one has unregistered is what caused the
    /// "Assertion failure. Value was True Expected: False" crash in
    /// TrackableSpawner.RegisterCreatedTrackable), then creates the new
    /// anchor through ARAnchorManager and instantiates the target as its
    /// child. This ID guard also makes sure that if HandleAnchorLost fires
    /// again while this coroutine is still awaiting the async anchor add,
    /// the stale result gets discarded instead of spawning two targets.
    /// </summary>
    private async Task SpawnTargetAtAsync(Vec3Dto spawnPointLocalSpace)
    {
        int requestId = ++_spawnRequestId;

        CleanupCurrentTarget();

        // Let Destroy() of the previous target/anchor actually flush before
        // we register a new one -- Destroy() is deferred to end-of-frame,
        // and creating a new anchor before the old one unregisters is what
        // caused the "Assertion failure. Value was True Expected: False"
        // crash in TrackableSpawner.RegisterCreatedTrackable.
        await Awaitable.NextFrameAsync();

        if (requestId != _spawnRequestId) return; // superseded while we waited

        Camera cam = ARSessionManager.Instance.ArCamera;
        _currentTargetLocalPosAtSpawn = spawnPointLocalSpace.ToVector3();
        Vector3 worldPos = cam.transform.TransformPoint(_currentTargetLocalPosAtSpawn);
        var pose = new Pose(worldPos, Quaternion.identity);

        Result<ARAnchor> result;
        try
        {
            result = await anchorManager.TryAddAnchorAsync(pose);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Anchor creation threw ({e.Message}) -- retrying in 1s");
            if (requestId == _spawnRequestId) Invoke(nameof(RetryRequestNextTarget), 1f);
            return;
        }

        if (requestId != _spawnRequestId)
        {
            // A newer spawn started while this anchor was being created --
            // dispose it immediately so it doesn't leak or collide later.
            if (result.status.IsSuccess()) Destroy(result.value.gameObject);
            return;
        }

        if (!result.status.IsSuccess())
        {
            Debug.LogWarning($"Anchor creation failed ({result.status}) -- retrying in 1s");
            Invoke(nameof(RetryRequestNextTarget), 1f);
            return;
        }

        _currentAnchor = result.value;
        _currentTarget = Instantiate(birdPrefab, _currentAnchor.transform);
        _currentTarget.transform.localPosition = Vector3.zero;
        _currentTarget.transform.localRotation = Quaternion.identity;

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

    /// <summary>
    /// Destroys the current target GameObject (which is now the child of
    /// its anchor, so this also tears down the ARAnchor) and stops the
    /// health-check coroutine. Does NOT create anything new -- callers
    /// that spawn a replacement must go through SpawnTargetAtRoutine,
    /// which waits a frame after calling this so the destroy actually
    /// flushes before a new anchor is registered.
    /// </summary>
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