using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

/// <summary>
/// Runs the star_collect exercise mode: spawns a batch of stars at once
/// (count comes from the backend's difficulty.target_count), lets the
/// patient collect them in any order, then submits all results together
/// and requests the next round. SessionManager delegates to this when
/// exercise_mode == "star_collect" and otherwise ignores it entirely.
/// </summary>
public class StarCollectRoundController : MonoBehaviour
{
    [SerializeField] private SafeSpawnPointFinder spawnFinder;
    [SerializeField] private GameObject starPrefab;
    [SerializeField] private HUDController hud;

    private string _sessionId;
    private readonly List<GameObject> _activeStarObjects = new List<GameObject>();
    private readonly List<StarTarget> _activeStars = new List<StarTarget>();
    private readonly List<TrialResultDto> _pendingResults = new List<TrialResultDto>();
    private int _collectedCount;
    private DifficultyParamsDto _currentDifficulty;
    private Coroutine _roundTimeoutRoutine;

    public void BeginSession(string sessionId)
    {
        _sessionId = sessionId;
        RequestAndSpawnRound(new List<TrialResultDto>());
    }

    private void RequestAndSpawnRound(List<TrialResultDto> lastRoundResults)
    {
        spawnFinder.RefreshCandidates();
        var candidates = spawnFinder.GetCandidatesInCameraLocalSpace();
        hud?.SetStatus("Requesting next round...");

        NetworkClient.Instance.RequestNextRound(_sessionId, candidates, lastRoundResults,
            onSuccess: (resp) =>
            {
                _currentDifficulty = resp.difficulty;
                SpawnRound(resp.spawn_points);
                hud?.SetDifficulty(resp.difficulty);
            },
            onError: (err) =>
            {
                Debug.LogWarning($"{err} -- retrying in 1s");
                Invoke(nameof(RetryLastRound), 1f);
            });
    }

    private List<TrialResultDto> _lastAttemptedResults = new List<TrialResultDto>();
    private void RetryLastRound() => RequestAndSpawnRound(_lastAttemptedResults);

    private void SpawnRound(List<Vec3Dto> spawnPointsLocal)
    {
        CleanupActiveStars();
        _pendingResults.Clear();
        _collectedCount = 0;

        Camera cam = ARSessionManager.Instance.ArCamera;

        foreach (var pointDto in spawnPointsLocal)
        {
            Vector3 localPos = pointDto.ToVector3();
            Vector3 worldPos = cam.transform.TransformPoint(localPos);

            GameObject star = Instantiate(starPrefab, worldPos, Quaternion.identity);
            star.AddComponent<ARAnchor>(); // fixed in real-world space, same reasoning as the bird's anchor

            var starTarget = star.GetComponent<StarTarget>();
            starTarget.Initialize(localPos);
            starTarget.OnCollected += HandleStarCollected;

            _activeStarObjects.Add(star);
            _activeStars.Add(starTarget);
        }

        hud?.SetStarProgress(0, _activeStars.Count);

        if (_roundTimeoutRoutine != null) StopCoroutine(_roundTimeoutRoutine);
        _roundTimeoutRoutine = StartCoroutine(RoundTimeout(_currentDifficulty.time_limit_s));
    }

    private void HandleStarCollected(StarTarget star, float reactionMs, float gazeAngleDeg)
    {
        _collectedCount++;
        hud?.SetStarProgress(_collectedCount, _activeStars.Count);

        _pendingResults.Add(new TrialResultDto
        {
            hit = true,
            reaction_time_ms = reactionMs,
            gaze_angle_deg = gazeAngleDeg,
            target_position = new Vec3Dto(star.SpawnLocalPos)
        });

        if (_collectedCount >= _activeStars.Count)
        {
            if (_roundTimeoutRoutine != null) StopCoroutine(_roundTimeoutRoutine);
            EndRound();
        }
    }

    private IEnumerator RoundTimeout(float seconds)
    {
        yield return new WaitForSeconds(Mathf.Max(1f, seconds));
        EndRound(); // any star not yet collected gets logged as a miss below
    }

    private void EndRound()
    {
        // Any star still active (not collected) at this point is a miss --
        // this covers both the timeout path and guards against double-calls.
        foreach (var star in _activeStars)
        {
            if (!star.IsCollected)
            {
                _pendingResults.Add(new TrialResultDto
                {
                    hit = false,
                    reaction_time_ms = star.TimeSinceSpawnMs(),
                    gaze_angle_deg = 0f,
                    target_position = new Vec3Dto(star.SpawnLocalPos)
                });
            }
        }

        hud?.SetLastResult(hit: _collectedCount == _activeStars.Count, reactionMs: 0);

        var resultsForThisRound = new List<TrialResultDto>(_pendingResults);
        _lastAttemptedResults = resultsForThisRound;
        RequestAndSpawnRound(resultsForThisRound);
    }

    private void CleanupActiveStars()
    {
        foreach (var star in _activeStars)
            star.OnCollected -= HandleStarCollected;
        foreach (var obj in _activeStarObjects)
            if (obj != null) Destroy(obj);
        _activeStarObjects.Clear();
        _activeStars.Clear();
    }

    private void OnDestroy()
    {
        CleanupActiveStars();
    }
}
