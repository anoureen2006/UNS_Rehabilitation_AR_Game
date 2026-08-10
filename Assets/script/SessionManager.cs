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

    [Header("Audio hint")]
    [Tooltip("Seconds of no hit before the target plays a spatialized call to help the patient find it. Set to 0 or a very large number to effectively disable.")]
    [SerializeField] private float hintDelayS = 15f;

    public GameState State { get; private set; } = GameState.Idle;

    private string _sessionId;
    private GameObject _currentTarget;
    private ARAnchor _currentAnchor;
    private DifficultyParamsDto _currentDifficulty;
    private Vector3 _currentTargetLocalPosAtSpawn; // camera-local space, for logging hemifield correctly
    private Coroutine _timeoutRoutine;
    private Coroutine _anchorHealthRoutine;
    private Coroutine _hintRoutine;

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

        // Pull menu-selected values if the player came from MainMenuController.
        // Falls back to the Inspector-configured defaults above so this scene
        // still works standalone (e.g. testing the game scene directly in
        // Editor without going through the menu first).
        if (PlayerPrefs.HasKey("usn_neglect_side"))
        {
            neglectSide = PlayerPrefs.GetString("usn_neglect_side", neglectSide);
            exerciseMode = PlayerPrefs.GetString("usn_exercise_mode", exerciseMode);
            patientId = PlayerPrefs.GetString("usn_patient_id", patientId);
        }
        hud?.SetStatus($"Patient: {patientId}  |  Neglect side: {neglectSide}  |  Mode: {exerciseMode}");

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

        Camera cam = ARSessionManager.Instance.ArCamera;
        _currentTargetLocalPosAtSpawn = spawnPointLocalSpace.ToVector3();
        Vector3 worldPos = cam.transform.TransformPoint(_currentTargetLocalPosAtSpawn);

        // If a bird is already on screen (i.e. this isn't the very first
        // spawn of the session), fly it smoothly to the new spot BEFORE
        // tearing it down -- by the time we destroy/recreate at the new
        // anchor below, the replacement appears exactly where the old one
        // just arrived, so the swap reads as one continuous flight instead
        // of a teleport.
        if (_currentTarget != null)
        {
            var oldBird = _currentTarget.GetComponent<BirdController>();
            if (oldBird != null) await oldBird.FlyToAsync(worldPos);
        }

        if (requestId != _spawnRequestId) return; // superseded while we were flying/waiting

        CleanupCurrentTarget();

        // Let Destroy() of the previous target/anchor actually flush before
        // we register a new one -- Destroy() is deferred to end-of-frame,
        // and creating a new anchor before the old one unregisters is what
        // caused the "Assertion failure. Value was True Expected: False"
        // crash in TrackableSpawner.RegisterCreatedTrackable.
        await Awaitable.NextFrameAsync();

        if (requestId != _spawnRequestId) return; // superseded while we waited

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

        // Deliberately NOT parented under _currentAnchor.transform. If AR
        // tracking degrades (camera briefly covered/occluded, poor
        // lighting, fast motion), the anchor's pose can jitter or drift as
        // the subsystem re-estimates it -- a bird parented under that
        // jittering transform visibly glitches even though BirdController's
        // own movement math is smooth. Instead we read the anchor's world
        // position ONCE here and instantiate the bird as an independent
        // root-level object; BirdController owns 100% of its motion from
        // this point on, regardless of what the AR subsystem later does to
        // the anchor. The anchor is kept only for MonitorAnchorHealth's
        // tracking-state checks (a real, sustained loss still correctly
        // triggers a respawn).
        _currentTarget = Instantiate(birdPrefab, worldPos, Quaternion.identity);

        var birdController = _currentTarget.GetComponent<BirdController>();
        // BirdController owns all wandering/orbit motion internally now --
        // just tell it where "home" is.
        birdController.SetTarget(worldPos, _currentDifficulty.speed);

        gazeTracker.BeginTrial(_currentTarget.transform);
        State = GameState.Tracking;

        gazeTracker.OnTargetAcquired += HandleTargetAcquired;

        if (_timeoutRoutine != null) StopCoroutine(_timeoutRoutine);
        _timeoutRoutine = StartCoroutine(MissTimeout(_currentDifficulty.time_limit_s));

        if (_anchorHealthRoutine != null) StopCoroutine(_anchorHealthRoutine);
        _anchorHealthRoutine = StartCoroutine(MonitorAnchorHealth());

        if (_hintRoutine != null) StopCoroutine(_hintRoutine);
        // Clamp to a safe fraction of THIS trial's actual time limit -- if
        // hintDelayS (e.g. 15s) is longer than the trial's time_limit_s
        // (from the difficulty controller), MissTimeout would end the
        // trial before the hint ever got a chance to play, silently
        // "breaking" the sound with no error anywhere.
        float effectiveHintDelay = Mathf.Min(hintDelayS, _currentDifficulty.time_limit_s * 0.6f);
        if (effectiveHintDelay > 0.1f) _hintRoutine = StartCoroutine(HintAfterDelay(effectiveHintDelay));
    }

    /// <summary>
    /// Waits delayS seconds, then asks the bird to play its spatialized
    /// call so the patient has an auditory cue toward the neglected side.
    /// BirdController itself guards against double-playing or playing after
    /// the target's already been found -- this coroutine just fires the ask.
    /// </summary>
    private IEnumerator HintAfterDelay(float delayS)
    {
        yield return new WaitForSeconds(delayS);
        if (_currentTarget != null)
        {
            var bird = _currentTarget.GetComponent<BirdController>();
            bird?.PlayHintCallIfNeeded();
        }
    }

    /// <summary>
    /// Periodically checks whether the current anchor is still being
    /// actively tracked by the AR subsystem. If the patient walks
    /// somewhere with no tracked geometry, moves too fast, covers the
    /// camera, or the tracking system loses confidence, the anchor's
    /// TrackingState degrades from Tracking to Limited/None. When that
    /// SUSTAINS for several consecutive checks (not just one blip) we
    /// respawn the target rather than let it float in a stale/wrong
    /// position -- and crucially this does NOT count as a miss, since
    /// it's a tracking failure, not a patient performance failure.
    ///
    /// The consecutive-check requirement (rather than reacting to the
    /// very first non-Tracking reading) exists specifically because a
    /// covered/occluded camera causes TrackingState to flicker rapidly
    /// between Tracking and Limited -- reacting instantly caused a
    /// destroy/recreate respawn on every flicker, which is what looked
    /// like constant "glitching."
    /// </summary>
    private IEnumerator MonitorAnchorHealth()
    {
        const int requiredConsecutiveBadChecks = 4; // ~2s sustained loss at the default 0.5s interval
        int badCheckStreak = 0;

        var wait = new WaitForSeconds(anchorHealthCheckIntervalS);

        // Brief grace period right after a fresh anchor is created --
        // newly created anchors can legitimately start as Limited for a
        // moment while the AR subsystem settles.
        yield return new WaitForSeconds(1f);

        while (_currentAnchor != null)
        {
            if (_currentAnchor.trackingState != TrackingState.Tracking)
            {
                badCheckStreak++;
                if (badCheckStreak >= requiredConsecutiveBadChecks)
                {
                    HandleAnchorLost();
                    yield break;
                }
            }
            else
            {
                badCheckStreak = 0; // any good reading resets the streak
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
        if (_hintRoutine != null) StopCoroutine(_hintRoutine);
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

        // Tell the bird the trial is over so it stops wandering/flying and
        // the Animator transitions to perched -- BirdController no longer
        // does this on its own via proximity (see BirdController header).
        if (_currentTarget != null)
        {
            _currentTarget.GetComponent<BirdController>()?.Stop();
        }

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
    /// <summary>
    /// Destroys the current target GameObject AND the anchor's own
    /// GameObject -- the bird is no longer parented under the anchor (see
    /// SpawnTargetAtAsync), so both need to be destroyed explicitly or the
    /// anchor leaks (stays alive, invisible, forever) every single round.
    /// Also stops the health-check/hint coroutines. Does NOT create
    /// anything new -- callers that spawn a replacement must go through
    /// SpawnTargetAtAsync, which waits a frame after calling this so the
    /// destroys actually flush before a new anchor is registered.
    /// </summary>
    private void CleanupCurrentTarget()
    {
        if (_anchorHealthRoutine != null) StopCoroutine(_anchorHealthRoutine);
        _anchorHealthRoutine = null;
        if (_hintRoutine != null) StopCoroutine(_hintRoutine);
        _hintRoutine = null;
        if (_currentTarget != null) Destroy(_currentTarget);
        if (_currentAnchor != null) Destroy(_currentAnchor.gameObject);
        _currentTarget = null;
        _currentAnchor = null;
    }

    private void OnDestroy()
    {
        if (gazeTracker != null) gazeTracker.OnTargetAcquired -= HandleTargetAcquired;
    }
}