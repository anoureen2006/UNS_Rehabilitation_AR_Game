using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

[Serializable]
public class Vec3Dto
{
    public float x;
    public float y;
    public float z;

    public Vec3Dto()
    {
    }

    public Vec3Dto(Vector3 v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    public Vector3 ToVector3()
    {
        return new Vector3(x, y, z);
    }
}

[Serializable]
public class DifficultyParamsDto
{
    public float speed;
    public float eccentricity_deg;
    public float distance_m;
    public int target_count;
    public float time_limit_s;
}

[Serializable]
public class SessionStartRequestDto
{
    public string patient_id;
    public string neglect_side;
    public string exercise_mode;
}

[Serializable]
public class SessionStartResponseDto
{
    public string session_id;
    public DifficultyParamsDto initial_difficulty;
}

[Serializable]
public class TrialResultDto
{
    public bool hit;
    public float reaction_time_ms;
    public float gaze_angle_deg;
    public Vec3Dto target_position;
}

[Serializable]
public class NextTargetRequestDto
{
    public List<Vec3Dto> candidate_points;
    public TrialResultDto last_result;
}

[Serializable]
public class NextTargetResponseDto
{
    public Vec3Dto spawn_point;
    public DifficultyParamsDto difficulty;

    // Backend should provide this if available.
    public float decision_time_ms;
}

[Serializable]
public class NextRoundRequestDto
{
    public List<Vec3Dto> candidate_points;
    public List<TrialResultDto> last_round_results;
}

[Serializable]
public class NextRoundResponseDto
{
    public List<Vec3Dto> spawn_points;
    public DifficultyParamsDto difficulty;

    public float decision_time_ms;
}

public class NetworkClient : MonoBehaviour
{
    public static NetworkClient Instance { get; private set; }

    [SerializeField]
    private string baseUrl =
        "http://192.168.1.42:8000";

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }

    // ============================================================
    // START SESSION
    // ============================================================

    public void StartSession(
        string patientId,
        string neglectSide,
        string exerciseMode,
        Action<SessionStartResponseDto> onSuccess,
        Action<string> onError
    )
    {
        var body =
            new SessionStartRequestDto
            {
                patient_id = patientId,
                neglect_side = neglectSide,
                exercise_mode = exerciseMode
            };

        string json =
            JsonUtility.ToJson(body);

        StartCoroutine(
            PostJson<SessionStartResponseDto>(
                $"{baseUrl}/session/start",
                json,
                onSuccess,
                onError
            )
        );
    }

    // ============================================================
    // NEXT TARGET
    // ============================================================

    public void RequestNextTarget(
        string sessionId,
        List<Vector3> candidatePointsLocalSpace,
        TrialResultDto lastResult,
        Action<NextTargetResponseDto> onSuccess,
        Action<string> onError
    )
    {
        var dto =
            new NextTargetRequestDto
            {
                candidate_points =
                    new List<Vec3Dto>(),

                last_result =
                    lastResult
            };

        if (candidatePointsLocalSpace != null)
        {
            foreach (
                Vector3 point
                in candidatePointsLocalSpace
            )
            {
                dto.candidate_points.Add(
                    new Vec3Dto(point)
                );
            }
        }

        string json =
            JsonUtility.ToJson(dto);

        Stopwatch apiTimer =
            Stopwatch.StartNew();

        Stopwatch e2eTimer =
            ARPerformanceMonitor.Instance
                ?.StartEndToEndLatency();

        StartCoroutine(
            PostJson<NextTargetResponseDto>(
                $"{baseUrl}/session/{sessionId}/next-target",
                json,

                response =>
                {
                    apiTimer.Stop();

                    float apiLatencyMs =
                        apiTimer.ElapsedTicks *
                        1000f /
                        Stopwatch.Frequency;

                    if (ARPerformanceMonitor.Instance != null)
                    {
                        ARPerformanceMonitor.Instance
                            .RecordApiLatency(
                                apiLatencyMs
                            );

                        ARPerformanceMonitor.Instance
                            .RecordAdaptiveDecisionTime(
                                response.decision_time_ms
                            );

                        ARPerformanceMonitor.Instance
                            .EndEndToEndLatency(
                                e2eTimer
                            );
                    }

                    onSuccess?.Invoke(response);
                },

                error =>
                {
                    apiTimer.Stop();

                    if (ARPerformanceMonitor.Instance != null)
                    {
                        ARPerformanceMonitor.Instance
                            .EndEndToEndLatency(
                                e2eTimer
                            );
                    }

                    onError?.Invoke(error);
                }
            )
        );
    }

    // ============================================================
    // NEXT ROUND
    // ============================================================

    public void RequestNextRound(
        string sessionId,
        List<Vector3> candidatePointsLocalSpace,
        List<TrialResultDto> lastRoundResults,
        Action<NextRoundResponseDto> onSuccess,
        Action<string> onError
    )
    {
        var dto =
            new NextRoundRequestDto
            {
                candidate_points =
                    new List<Vec3Dto>(),

                last_round_results =
                    lastRoundResults ??
                    new List<TrialResultDto>()
            };

        if (candidatePointsLocalSpace != null)
        {
            foreach (
                Vector3 point
                in candidatePointsLocalSpace
            )
            {
                dto.candidate_points.Add(
                    new Vec3Dto(point)
                );
            }
        }

        string json =
            JsonUtility.ToJson(dto);

        Stopwatch apiTimer =
            Stopwatch.StartNew();

        Stopwatch e2eTimer =
            ARPerformanceMonitor.Instance
                ?.StartEndToEndLatency();

        StartCoroutine(
            PostJson<NextRoundResponseDto>(
                $"{baseUrl}/session/{sessionId}/next-round",
                json,

                response =>
                {
                    apiTimer.Stop();

                    float apiLatencyMs =
                        apiTimer.ElapsedTicks *
                        1000f /
                        Stopwatch.Frequency;

                    if (ARPerformanceMonitor.Instance != null)
                    {
                        ARPerformanceMonitor.Instance
                            .RecordApiLatency(
                                apiLatencyMs
                            );

                        ARPerformanceMonitor.Instance
                            .RecordAdaptiveDecisionTime(
                                response.decision_time_ms
                            );

                        ARPerformanceMonitor.Instance
                            .EndEndToEndLatency(
                                e2eTimer
                            );
                    }

                    onSuccess?.Invoke(response);
                },

                error =>
                {
                    apiTimer.Stop();

                    if (ARPerformanceMonitor.Instance != null)
                    {
                        ARPerformanceMonitor.Instance
                            .EndEndToEndLatency(
                                e2eTimer
                            );
                    }

                    onError?.Invoke(error);
                }
            )
        );
    }

    // ============================================================
    // GENERIC POST JSON
    // ============================================================

    private IEnumerator PostJson<TResponse>(
        string url,
        string jsonBody,
        Action<TResponse> onSuccess,
        Action<string> onError
    )
    {
        using (
            UnityWebRequest req =
                new UnityWebRequest(
                    url,
                    UnityWebRequest.kHttpVerbPOST
                )
        )
        {
            byte[] bodyRaw =
                Encoding.UTF8.GetBytes(
                    jsonBody
                );

            req.uploadHandler =
                new UploadHandlerRaw(
                    bodyRaw
                );

            req.downloadHandler =
                new DownloadHandlerBuffer();

            req.SetRequestHeader(
                "Content-Type",
                "application/json"
            );

            req.timeout = 8;

            yield return req.SendWebRequest();

            bool failed =
                req.result !=
                UnityWebRequest.Result.Success;

            if (failed)
            {
                onError?.Invoke(
                    $"{url} failed: " +
                    $"{req.error} " +
                    $"(HTTP {req.responseCode})"
                );

                yield break;
            }

            string raw =
                req.downloadHandler.text;

            if (string.IsNullOrWhiteSpace(raw))
            {
                onError?.Invoke(
                    $"Empty response from {url}"
                );

                yield break;
            }

            TResponse parsed;

            try
            {
                parsed =
                    JsonUtility.FromJson<TResponse>(
                        raw
                    );
            }
            catch (Exception e)
            {
                onError?.Invoke(
                    "Failed to parse response from " +
                    url +
                    ": " +
                    e.Message +
                    "\nRaw: " +
                    raw
                );

                yield break;
            }

            if (parsed == null)
            {
                onError?.Invoke(
                    $"Response parsing returned null from {url}" +
                    $"\nRaw: {raw}"
                );

                yield break;
            }

            onSuccess?.Invoke(parsed);
        }
    }
}