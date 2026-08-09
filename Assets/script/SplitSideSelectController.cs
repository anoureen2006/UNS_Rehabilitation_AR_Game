using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using TMPro;

/// <summary>
/// Split-screen side-selection screen. The full screen is divided into a
/// left half and a right half, each a large tappable panel. The patient
/// taps whichever half they can actually see -- the OTHER half is then set
/// as the neglected side. This is deliberately not framed as "which side
/// is affected" (a patient with USN often can't reliably judge that about
/// themselves); it's framed as "what can you see," which is the thing
/// they CAN actually answer, and the neglect side is inferred from that.
///
/// Handoff to the game scene is via PlayerPrefs, read once by
/// SessionManager.Start() (see usn_neglect_side / usn_exercise_mode /
/// usn_patient_id keys).
///
/// Scene setup:
///   Canvas (Screen Space - Overlay)
///     - LeftPanel   (Image + Button, anchored to left half, full height)
///     - RightPanel  (Image + Button, anchored to right half, full height)
///     - InstructionText (TMP, centered, e.g. "Tap the side of the screen you can see")
///     - ConfirmOverlay (initially inactive) -> ConfirmLabel (TMP) + ConfirmButton
/// Attach this script to an empty "SplitScreenManager" GameObject.
/// </summary>
public class SplitSideSelectController : MonoBehaviour
{
    [Header("Split panels")]
    [SerializeField] private Button leftPanelButton;
    [SerializeField] private Button rightPanelButton;
    [SerializeField] private Image leftPanelImage;
    [SerializeField] private Image rightPanelImage;

    [Header("Instruction / feedback")]
    [SerializeField] private TextMeshProUGUI instructionText; // "Tap the side of the screen you can see"
    [SerializeField] private Color unselectedColor = new Color(0.15f, 0.15f, 0.18f);
    [SerializeField] private Color selectedColor = new Color(0.16f, 0.45f, 0.35f);   // the side they said they CAN see
    [SerializeField] private Color neglectedColor = new Color(0.5f, 0.15f, 0.15f);   // the inferred neglected side

    [Header("Confirmation step")]
    [Tooltip("Shown after a side is tapped, so a clinician/researcher can confirm before starting -- prevents an accidental tap from silently deciding the session.")]
    [SerializeField] private GameObject confirmOverlay;
    [SerializeField] private TextMeshProUGUI confirmLabel; // "Neglected side: LEFT -- start session?"
    [SerializeField] private Button confirmButton;
    [SerializeField] private Button cancelButton; // lets them re-tap if wrong

    [Header("Exercise mode")]
    [Tooltip("Optional -- if null, defaults to bird_chase. Keeping this off the split-screen itself so the screen stays purely about the left/right task.")]
    [SerializeField] private TMP_Dropdown modeDropdown;

    [Header("Start")]
    [SerializeField] private string gameSceneName = "GameScene";
    [SerializeField] private TMP_InputField patientIdField; // optional

    private string _neglectedSide = null; // "left" | "right"

    private void Start()
    {
        leftPanelButton.onClick.AddListener(() => OnSideTapped(canSeeSide: "left"));
        rightPanelButton.onClick.AddListener(() => OnSideTapped(canSeeSide: "right"));
        confirmButton.onClick.AddListener(OnConfirmPressed);
        cancelButton.onClick.AddListener(ResetSelection);

        confirmOverlay.SetActive(false);
        ResetSelection();
    }

    private void OnSideTapped(string canSeeSide)
    {
        _neglectedSide = canSeeSide == "left" ? "right" : "left";

        // Color the tapped side as "confirmed visible", the other as "neglected" --
        // gives an immediate, unambiguous visual read for anyone watching (judges included).
        leftPanelImage.color = canSeeSide == "left" ? selectedColor : neglectedColor;
        rightPanelImage.color = canSeeSide == "right" ? selectedColor : neglectedColor;

        confirmLabel.text = $"Neglected side: {_neglectedSide.ToUpper()}";
        confirmOverlay.SetActive(true);
    }

    private void ResetSelection()
    {
        _neglectedSide = null;
        leftPanelImage.color = unselectedColor;
        rightPanelImage.color = unselectedColor;
        instructionText.text = "Tap the side of the screen you can see";
        confirmOverlay.SetActive(false);
    }

    private void OnConfirmPressed()
    {
        if (_neglectedSide == null) return;

        string exerciseMode = (modeDropdown != null && modeDropdown.value == 1) ? "star_collect" : "bird_chase";
        string patientId = string.IsNullOrWhiteSpace(patientIdField != null ? patientIdField.text : null)
            ? $"patient_{System.DateTime.Now:yyyyMMdd_HHmmss}"
            : patientIdField.text.Trim();

        PlayerPrefs.SetString("usn_neglect_side", _neglectedSide);
        PlayerPrefs.SetString("usn_exercise_mode", exerciseMode);
        PlayerPrefs.SetString("usn_patient_id", patientId);
        PlayerPrefs.Save();

        SceneManager.LoadScene(gameSceneName);
    }
}
