using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class StarCollectRoundController : MonoBehaviour
{
    [SerializeField] private SafeSpawnPointFinder spawnFinder;
    [SerializeField] private GameObject starPrefab;
    [SerializeField] private HUDController hud;

    [Header("AR")]
    [SerializeField] private ARAnchorManager anchorManager;

    private string _sessionId;

    private readonly List<GameObject>
        _activeStarObjects =
            new List<GameObject>();

    private readonly List<StarTarget>
        _activeStars =
            new List<StarTarget>();

    private readonly List<ARAnchor>
        _activeAnchors =
            new List<ARAnchor>();

    private readonly List<TrialResultDto>
        _pendingResults =
            new List<TrialResultDto>();

    private int _collectedCount;

    private DifficultyParamsDto
        _currentDifficulty;

    private Coroutine
        _roundTimeoutRoutine;

    private bool _roundEnding;

    private List<TrialResultDto>
        _lastAttemptedResults =
            new List<TrialResultDto>();

    public void BeginSession(
        string sessionId
    )
    {
        _sessionId = sessionId;

        if (anchorManager == null)
        {
            anchorManager =
                ARSessionManager.Instance
                    ?.AnchorManager;
        }

        RequestAndSpawnRound(
            new List<TrialResultDto>()
        );
    }

    private void RequestAndSpawnRound(
        List<TrialResultDto>
            lastRoundResults
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

        if (spawnFinder == null)
        {
            Debug.LogError(
                "SafeSpawnPointFinder missing."
            );

            return;
        }

        spawnFinder.RefreshCandidates();

        List<Vector3> candidates =
            spawnFinder
                .GetCandidatesInCameraLocalSpace();

        hud?.SetStatus(
            "Requesting next round..."
        );

        NetworkClient.Instance
            .RequestNextRound(
                _sessionId,
                candidates,
                lastRoundResults,

                resp =>
                {
                    _currentDifficulty =
                        resp.difficulty;

                    hud?.SetDifficulty(
                        resp.difficulty
                    );

                    _ =
                        SpawnRoundAsync(
                            resp.spawn_points
                        );
                },

                err =>
                {
                    Debug.LogWarning(
                        $"{err} -- retrying in 1s"
                    );

                    CancelInvoke(
                        nameof(
                            RetryLastRound
                        )
                    );

                    Invoke(
                        nameof(
                            RetryLastRound
                        ),
                        1f
                    );
                }
            );
    }

    private void RetryLastRound()
    {
        RequestAndSpawnRound(
            _lastAttemptedResults
        );
    }

    // ============================================================
    // SPAWN ROUND
    // ============================================================

    private async Task SpawnRoundAsync(
        List<Vec3Dto> spawnPointsLocal
    )
    {
        _roundEnding = false;

        CleanupActiveStars();

        _pendingResults.Clear();

        _collectedCount = 0;

        if (
            spawnPointsLocal == null ||
            spawnPointsLocal.Count == 0
        )
        {
            Debug.LogWarning(
                "Backend returned no spawn points."
            );

            Invoke(
                nameof(
                    RetryLastRound
                ),
                1f
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

        Camera cam =
            ARSessionManager.Instance?.ArCamera;

        if (cam == null)
        {
            Debug.LogError(
                "AR Camera missing."
            );

            return;
        }

        foreach (
            Vec3Dto pointDto
            in spawnPointsLocal
        )
        {
            if (pointDto == null)
                continue;

            Vector3 localPos =
                pointDto.ToVector3();

            Vector3 worldPos =
                cam.transform.TransformPoint(
                    localPos
                );

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
                    $"Star anchor failed: {e.Message}"
                );

                continue;
            }

            if (
                !result.status.IsSuccess() ||
                result.value == null
            )
            {
                Debug.LogWarning(
                    $"Star anchor failed: {result.status}"
                );

                continue;
            }

            ARAnchor anchor =
                result.value;

            _activeAnchors.Add(
                anchor
            );

            GameObject star =
                Instantiate(
                    starPrefab,
                    worldPos,
                    Quaternion.identity
                );

            // Star is NOT parented under anchor.
            // The anchor is only used for AR tracking/reference.
            _activeStarObjects.Add(
                star
            );

            StarTarget starTarget =
                star.GetComponent<StarTarget>();

            if (starTarget == null)
            {
                Debug.LogError(
                    "Star prefab does not contain StarTarget."
                );

                Destroy(star);

                continue;
            }

            starTarget.Initialize(
                localPos
            );

            starTarget.OnCollected +=
                HandleStarCollected;

            _activeStars.Add(
                starTarget
            );
        }

        hud?.SetStarProgress(
            0,
            _activeStars.Count
        );

        if (_activeStars.Count == 0)
        {
            Debug.LogWarning(
                "No valid stars were spawned."
            );

            Invoke(
                nameof(
                    RetryLastRound
                ),
                1f
            );

            return;
        }

        if (_roundTimeoutRoutine != null)
        {
            StopCoroutine(
                _roundTimeoutRoutine
            );
        }

        _roundTimeoutRoutine =
            StartCoroutine(
                RoundTimeout(
                    _currentDifficulty.time_limit_s
                )
            );
    }

    // ============================================================
    // STAR COLLECTED
    // ============================================================

    private void HandleStarCollected(
        StarTarget star,
        float reactionMs,
        float gazeAngleDeg
    )
    {
        if (_roundEnding)
            return;

        if (star == null)
            return;

        if (star.IsCollected == false)
        {
            // Event should only fire after collection.
            return;
        }

        _collectedCount++;

        hud?.SetStarProgress(
            _collectedCount,
            _activeStars.Count
        );

        _pendingResults.Add(
            new TrialResultDto
            {
                hit = true,
                reaction_time_ms =
                    reactionMs,
                gaze_angle_deg =
                    gazeAngleDeg,
                target_position =
                    new Vec3Dto(
                        star.SpawnLocalPos
                    )
            }
        );

        if (
            _collectedCount >=
            _activeStars.Count
        )
        {
            EndRound();
        }
    }

    // ============================================================
    // TIMEOUT
    // ============================================================

    private IEnumerator RoundTimeout(
        float seconds
    )
    {
        yield return new WaitForSeconds(
            Mathf.Max(
                1f,
                seconds
            )
        );

        EndRound();
    }

    // ============================================================
    // END ROUND
    // ============================================================

    private void EndRound()
    {
        if (_roundEnding)
            return;

        _roundEnding = true;

        if (_roundTimeoutRoutine != null)
        {
            StopCoroutine(
                _roundTimeoutRoutine
            );

            _roundTimeoutRoutine = null;
        }

        foreach (
            StarTarget star
            in _activeStars
        )
        {
            if (
                star == null ||
                star.IsCollected
            )
            {
                continue;
            }

            _pendingResults.Add(
                new TrialResultDto
                {
                    hit = false,
                    reaction_time_ms =
                        star.TimeSinceSpawnMs(),
                    gaze_angle_deg = 0f,
                    target_position =
                        new Vec3Dto(
                            star.SpawnLocalPos
                        )
                }
            );
        }

        hud?.SetLastResult(
            _collectedCount ==
            _activeStars.Count,
            0f
        );

        List<TrialResultDto>
            resultsForThisRound =
                new List<TrialResultDto>(
                    _pendingResults
                );

        _lastAttemptedResults =
            resultsForThisRound;

        RequestAndSpawnRound(
            resultsForThisRound
        );
    }

    // ============================================================
    // CLEANUP
    // ============================================================

    private void CleanupActiveStars()
    {
        foreach (
            StarTarget star
            in _activeStars
        )
        {
            if (star != null)
            {
                star.OnCollected -=
                    HandleStarCollected;
            }
        }

        foreach (
            GameObject obj
            in _activeStarObjects
        )
        {
            if (obj != null)
                Destroy(obj);
        }

        foreach (
            ARAnchor anchor
            in _activeAnchors
        )
        {
            if (anchor != null)
                Destroy(anchor.gameObject);
        }

        _activeStarObjects.Clear();
        _activeStars.Clear();
        _activeAnchors.Clear();
    }

    private void OnDestroy()
    {
        CleanupActiveStars();
    }
}