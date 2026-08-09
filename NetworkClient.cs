using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

// ---- Data transfer objects -------------------------------------------
// Field names deliberately match the Python backend's JSON keys exactly
// (snake_case) since JsonUtility maps by field name. Do not rename these
// without also updating backend/schemas.py.

[Serializable]
public class Vec3Dto
{
    public float x, y, z;
    public Vec3Dto() { }
    public Vec3Dto(Vector3 v) { x = v.x; y = v.y; z = v.z; }
    public Vector3 ToVector3() => new Vector3(x, y, z);
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
    public string neglect_side;   // "left" or "right"
    public string exercise_mode;  // "bird_chase" or "star_collect"
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

// JsonUtility can't serialize a top-level List<T> directly, so wrap it.
[Serializable]
public class Vec3ListWrapper
{
    public List<Vec3Dto> items = new List<Vec3Dto>();
}

[Serializable]
public class NextTargetRequestDto
{
    public List<Vec3Dto> candidate_points;
    public TrialResultDto last_result; // null on the very first call after start
}

[Serializable]
public class NextTargetResponseDto
{
    public Vec3Dto spawn_point;
    public DifficultyParamsDto difficulty;
}

[Serializable]
public class TrialResultListWrapper
{
    public List<TrialResultDto> items = new List<TrialResultDto>();
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
}

/// <summary>
/// Thin async wrapper around UnityWebRequest. Every method here corresponds
/// to ONE discrete game event (session start, hit/miss + next-target
/// request) -- never called on a per-frame timer. See SessionManager for
/// where these get invoked.
/// </summary>
public class NetworkClient : MonoBehaviour
{
    public static NetworkClient Instance { get; private set; }

    [Tooltip("e.g. http://192.168.1.42:8000 -- your laptop's LAN IP, NOT localhost, if testing on a physical device.")]
    [SerializeField] private string baseUrl = "http://192.168.1.42:8000";

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    public void StartSession(string patientId, string neglectSide, string exerciseMode,
        Action<SessionStartResponseDto> onSuccess, Action<string> onError)
    {
        var body = new SessionStartRequestDto
        {
            patient_id = patientId,
            neglect_side = neglectSide,
            exercise_mode = exerciseMode
        };
        StartCoroutine(PostJson($"{baseUrl}/session/start", JsonUtility.ToJson(body), onSuccess, onError));
    }

    public void RequestNextTarget(string sessionId, List<Vector3> candidatePointsLocalSpace,
        TrialResultDto lastResult, Action<NextTargetResponseDto> onSuccess, Action<string> onError)
    {
        var dto = new NextTargetRequestDto
        {
            candidate_points = new List<Vec3Dto>(),
            last_result = lastResult
        };
        foreach (var p in candidatePointsLocalSpace)
            dto.candidate_points.Add(new Vec3Dto(p));

        string json = JsonUtility.ToJson(dto);
        StartCoroutine(PostJson($"{baseUrl}/session/{sessionId}/next-target", json, onSuccess, onError));
    }

    public void RequestNextRound(string sessionId, List<Vector3> candidatePointsLocalSpace,
        List<TrialResultDto> lastRoundResults, Action<NextRoundResponseDto> onSuccess, Action<string> onError)
    {
        var dto = new NextRoundRequestDto
        {
            candidate_points = new List<Vec3Dto>(),
            last_round_results = lastRoundResults ?? new List<TrialResultDto>()
        };
        foreach (var p in candidatePointsLocalSpace)
            dto.candidate_points.Add(new Vec3Dto(p));

        string json = JsonUtility.ToJson(dto);
        StartCoroutine(PostJson($"{baseUrl}/session/{sessionId}/next-round", json, onSuccess, onError));
    }

    private IEnumerator PostJson<TResponse>(string url, string jsonBody,
        Action<TResponse> onSuccess, Action<string> onError)
    {
        using (var req = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonBody);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = 8; // seconds -- fail fast rather than hang the game loop

            yield return req.SendWebRequest();

#if UNITY_2020_2_OR_NEWER
            bool failed = req.result != UnityWebRequest.Result.Success;
#else
            bool failed = req.isNetworkError || req.isHttpError;
#endif
            if (failed)
            {
                onError?.Invoke($"{url} failed: {req.error} (HTTP {req.responseCode})");
                yield break;
            }

            try
            {
                TResponse parsed = JsonUtility.FromJson<TResponse>(req.downloadHandler.text);
                onSuccess?.Invoke(parsed);
            }
            catch (Exception e)
            {
                onError?.Invoke($"Failed to parse response from {url}: {e.Message}\nRaw: {req.downloadHandler.text}");
            }
        }
    }
}
