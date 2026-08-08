using UnityEngine;

/// <summary>
/// Purely procedural idle motion (bob + gentle rotation wobble) so the
/// target doesn't look like a frozen placeholder, without requiring you to
/// build a real Animator Controller asset in the Editor. Attach to the same
/// bird prefab that has BirdController -- they don't need to know about
/// each other, this just adds a small local offset on top of whatever
/// BirdController is doing.
///
/// If/when you get a real animated bird model with its own Animator
/// Controller, just remove this component -- BirdController already has an
/// optional `animator` field ready for that instead.
/// </summary>
public class BirdIdleAnimator : MonoBehaviour
{
    [SerializeField] private float bobAmplitudeM = 0.02f;
    [SerializeField] private float bobFrequencyHz = 1.5f;
    [SerializeField] private float wobbleAmplitudeDeg = 6f;
    [SerializeField] private float wobbleFrequencyHz = 1.0f;

    private Vector3 _basePosition;
    private float _phaseOffset;

    private void Start()
    {
        _basePosition = transform.localPosition;
        // Randomize phase so multiple birds/stars on screen don't bob in
        // perfect unison -- small detail but noticeably less robotic.
        _phaseOffset = Random.Range(0f, Mathf.PI * 2f);
    }

    private void LateUpdate()
    {
        // LateUpdate so this layers on top of BirdController's Update()
        // movement rather than fighting it.
        float t = Time.time;
        float bob = Mathf.Sin((t * bobFrequencyHz * Mathf.PI * 2f) + _phaseOffset) * bobAmplitudeM;
        transform.position += Vector3.up * bob * Time.deltaTime * 60f * 0.0166f; // small per-frame nudge, not a hard position overwrite

        float wobble = Mathf.Sin((t * wobbleFrequencyHz * Mathf.PI * 2f) + _phaseOffset) * wobbleAmplitudeDeg;
        transform.rotation *= Quaternion.Euler(0f, 0f, wobble * Time.deltaTime);
    }
}
