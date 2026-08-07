using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Minimal debug/demo HUD. Attach to a Canvas and assign the Text fields
/// (regular UI.Text is fine for Day 1 -- swap to TextMeshPro later if you
/// want nicer fonts, the logic doesn't change).
/// </summary>
public class HUDController : MonoBehaviour
{
    [SerializeField] private Text statusText;
    [SerializeField] private Text difficultyText;
    [SerializeField] private Text lastResultText;
    [SerializeField] private Text starProgressText; // only used in star_collect mode, leave unassigned for bird_chase-only builds

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
