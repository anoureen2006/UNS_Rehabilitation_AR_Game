using UnityEngine;
using TMPro;

/// <summary>
/// Minimal debug/demo HUD. Attach to a Canvas and assign the TextMeshPro
/// fields below by dragging each corresponding TextMeshProUGUI object from
/// the Hierarchy.
/// </summary>
public class HUDController : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI statusText;
    [SerializeField] private TextMeshProUGUI difficultyText;
    [SerializeField] private TextMeshProUGUI lastResultText;
    [SerializeField] private TextMeshProUGUI starProgressText; // only used in star_collect mode, leave unassigned for bird_chase-only builds

    public void SetStarProgress(int collected, int total)
    {
        if (starProgressText != null)
            starProgressText.text = $"Stars: {collected} / {total}";
    }

    public void SetStatus(string message)
    {
        if (statusText != null) statusText.text = message;
    }

    public void SetDifficulty(DifficultyParamsDto d)
    {
        if (difficultyText != null)
        {
            difficultyText.text =
                $"speed {d.speed:F2}  |  eccentricity {d.eccentricity_deg:F0}°  |  distance {d.distance_m:F2}m";
        }
    }

    public void SetLastResult(bool hit, float reactionMs)
    {
        if (lastResultText != null)
        {
            lastResultText.text = hit
                ? $"HIT  ({reactionMs:F0} ms)"
                : $"MISS (timed out)";
            lastResultText.color = hit ? Color.green : Color.red;
        }
    }
}