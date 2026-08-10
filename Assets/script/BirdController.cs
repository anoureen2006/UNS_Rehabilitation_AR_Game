using UnityEngine;
using System.Threading.Tasks;

/// <summary>
/// Moves the bird (or star, or any target) GameObject with smooth,
/// continuous motion -- either gentle orbiting around a spawn point, or a
/// deliberate glide to a new destination. Attach to the target prefab
/// itself. SessionManager calls SetTarget()/FlyToAsync()/Stop(); this
/// script never talks to the network directly.
///
/// IMPORTANT SETUP STEP: on this GameObject's Animator component, make
/// sure "Apply Root Motion" is UNCHECKED. If it's left on, the Animator
/// will overwrite this script's position/rotation every frame using
/// whatever motion is baked into the animation clips, which looks exactly
/// like "rotation offset does nothing" / "movement is jittery" no matter
/// what values you tune here. This script intentionally does all its work
/// in LateUpdate (Unity evaluates Animator, including root motion, between
/// Update and LateUpdate) so it wins even if root motion is accidentally
/// left on -- but disabling it properly is still the correct fix and
/// avoids wasted overwrite work.
///
/// Also intentionally NOT parented under the AR anchor transform (see
/// SessionManager) -- the anchor's pose can jitter when AR tracking
/// degrades (e.g. camera briefly covered/occluded), and a bird parented
/// under a jittering transform visibly glitches even though this script's
/// own math is smooth. The anchor is only used once, to read the initial
/// spawn position; after that this script owns 100% of the bird's motion
/// in world space, independent of anything the AR subsystem does to the
/// anchor afterward.
///
/// Animator parameter mapping (Living Birds asset controller):
///   "flying"  (bool)  -- true for the entire time this target is on
///                         screen (orbiting OR gliding to a new spot).
///   "perched" (bool)  -- logical inverse of "flying". Only becomes true
///                         when SessionManager explicitly calls Stop().
///   "flyingDirectionX" / "flyingDirectionY" (float) -- normalized local
///                         movement direction, fed each frame, purely for
///                         banking/lean visual polish in the blend tree.
/// There is no "IsMoving" parameter in this controller -- do not reintroduce it.
/// </summary>
public class BirdController : MonoBehaviour
{
    private enum Motion { Idle, Orbiting, Gliding }

    [Header("Wander behavior")]
    [Tooltip("Radius of the smooth circular path the bird orbits around its spawn point.")]
    [SerializeField] private float wanderRadiusM = 0.15f;
    [Tooltip("Small vertical bob amplitude layered on top of the orbit, in meters.")]
    [SerializeField] private float verticalBobM = 0.03f;

    [Header("Orientation")]
    [Tooltip("Degrees to rotate the model's facing direction around the Y axis so the HEAD (not tail) leads movement. Tune this live in Play mode -- try 180 first (matches the clockwise orbit direction), then 0, 90, -90 if that's not quite right. If it STILL looks wrong at every value, the problem is Apply Root Motion being enabled on the Animator (see class header), not this value.")]
    [SerializeField] private float forwardAxisOffsetDeg = 0f;

    [Tooltip("If the bird still looks like it's flying backward no matter what forwardAxisOffsetDeg is set to, the likely cause is NOT root rotation -- it's flyingDirectionX/Y driving Living Birds' internal blend tree toward a backward-facing flap clip. Uncheck this to stop feeding those floats entirely as a diagnostic: if the backward look disappears, the fix belongs in the Animator's blend tree (check its axis/sign convention), not in this script.")]
    [SerializeField] private bool feedDirectionFloatsToAnimator = false;

    [Header("Animation")]
    [SerializeField] private Animator animator; // optional, leave null if no animation yet

    [Header("Audio hint")]
    [Tooltip("Played once if the patient hasn't found the target after some time -- see SessionManager's hint delay. Requires an AudioSource on this GameObject with Spatial Blend = 1 (3D) so direction is perceivable through headphones/stereo speakers.")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip[] hintCallClips; // pick randomly for variety across trials

    private Motion _motion = Motion.Idle;

    // Orbiting state
    private Vector3 _homeWorldPos;
    private float _orbitAngleRad;
    private float _phaseOffset;
    private float _speed = 0.5f;

    // Gliding state (FlyToAsync)
    private Vector3 _glideStartPos;
    private Vector3 _glideDestPos;
    private float _glideDurationS;
    private float _glideElapsedS;
    private TaskCompletionSource<bool> _glideCompletion;

    private bool _hintPlayedForCurrentTarget = false;

    public bool HasArrived { get; private set; }

    private void Awake()
    {
        if (animator != null && animator.applyRootMotion)
        {
            Debug.LogWarning($"{name}: Animator has 'Apply Root Motion' ENABLED. " +
                "This will fight BirdController for control of position/rotation every frame. " +
                "Uncheck it on the Animator component in the Inspector.", this);
        }
    }

    /// <summary>
    /// worldPosition is the point the bird should hover/orbit around.
    /// speed only affects how briskly it orbits, not a one-shot travel
    /// distance -- there is no "arrival" from orbiting; it continues until
    /// Stop() or FlyToAsync() is called.
    /// </summary>
    public void SetTarget(Vector3 worldPosition, float speed)
    {
        _homeWorldPos = worldPosition;
        _speed = Mathf.Max(0.05f, speed);
        HasArrived = false;
        _hintPlayedForCurrentTarget = false;

        _orbitAngleRad = 0f;
        _phaseOffset = Random.Range(0f, Mathf.PI * 2f);
        transform.position = ComputeOrbitPosition(_orbitAngleRad); // snap onto the orbit path immediately, no initial pop
        _motion = Motion.Orbiting;
        SetFlying(true);
    }

    /// <summary>
    /// Smoothly glides from the current position to a new destination,
    /// instead of teleporting there. Call this BEFORE destroying/replacing
    /// this GameObject at a new AR anchor -- by the time the swap happens,
    /// this bird is already sitting at the new spot, so the visible
    /// replacement looks like one continuous flight instead of a jump.
    /// </summary>
    public Task FlyToAsync(Vector3 destinationWorldPos, float durationS = 0.8f)
    {
        _glideStartPos = transform.position;
        _glideDestPos = destinationWorldPos;
        _glideDurationS = Mathf.Max(0.05f, durationS);
        _glideElapsedS = 0f;
        _motion = Motion.Gliding;
        SetFlying(true);

        _glideCompletion = new TaskCompletionSource<bool>();
        return _glideCompletion.Task;
    }

    /// <summary>
    /// Called by SessionManager when the trial actually resolves (hit,
    /// miss, anchor lost, or cleanup) -- this, not proximity, is what ends
    /// the flying animation. Safe to call mid-glide; releases anything
    /// awaiting FlyToAsync immediately.
    /// </summary>
    public void Stop()
    {
        _motion = Motion.Idle;
        HasArrived = true;
        SetFlying(false);

        if (_glideCompletion != null && !_glideCompletion.Task.IsCompleted)
            _glideCompletion.SetResult(true);
    }

    private Vector3 ComputeOrbitPosition(float angleRad)
    {
        float x = Mathf.Cos(angleRad + _phaseOffset) * wanderRadiusM;
        float z = Mathf.Sin(angleRad + _phaseOffset) * wanderRadiusM;
        float y = Mathf.Sin(Time.time * 1.3f + _phaseOffset) * verticalBobM;
        return _homeWorldPos + new Vector3(x, y, z);
    }

    /// <summary>
    /// LateUpdate, not Update -- Unity evaluates the Animator (including
    /// any root motion) between Update and LateUpdate, so doing all
    /// position/rotation work here guarantees this script always has the
    /// final say for the frame, even if Apply Root Motion is accidentally
    /// left enabled. This is also why FlyToAsync's glide progress is
    /// advanced here rather than in its own async loop -- one single
    /// source of truth for movement, every frame, same timing every time.
    /// </summary>
    private void LateUpdate()
{
    switch (_motion)
    {
        case Motion.Orbiting:
        {
            // Negative sign = Clockwise orbit
            float angularSpeedRadPerSec = -(_speed / Mathf.Max(0.02f, wanderRadiusM));
            _orbitAngleRad += angularSpeedRadPerSec * Time.deltaTime;

            Vector3 previousPos = transform.position;
            Vector3 nextPos = ComputeOrbitPosition(_orbitAngleRad);
            
            // Pass movement delta to unified movement handler
            ApplyMovement(previousPos, nextPos);
            break;
        }
        case Motion.Gliding:
        {
            _glideElapsedS += Time.deltaTime;
            float t = Mathf.Clamp01(_glideElapsedS / _glideDurationS);

            Vector3 previousPos = transform.position;
            Vector3 nextPos = Vector3.Lerp(_glideStartPos, _glideDestPos, t);
            ApplyMovement(previousPos, nextPos);

            if (t >= 1f)
            {
                _motion = Motion.Idle;
                _glideCompletion?.SetResult(true);
            }
            break;
        }
        case Motion.Idle:
        default:
            break;
    }
}

private void ApplyMovement(Vector3 previousPos, Vector3 nextPos)
{
    transform.position = nextPos;

    Vector3 dir = nextPos - previousPos;
    dir.y = 0f; // Ignore vertical bobbing for rotation math
    if (dir.sqrMagnitude <= 0.000001f) return;

    Vector3 forwardDir = dir.normalized;

    // 1. Set pure transform rotation facing the direction of travel
    // If the mesh is backwards, flip forwardDir to -forwardDir on this line:
    transform.rotation = Quaternion.LookRotation(forwardDir, Vector3.up);

    // 2. Prevent Living Birds Blend Tree from playing backward flap clips
    if (animator != null && feedDirectionFloatsToAnimator)
    {
        Vector3 localDir = transform.InverseTransformDirection(forwardDir);
        
        // Clamp Y parameter to positive values so Animator stays in 'Forward Flap' state
        animator.SetFloat("flyingDirectionX", localDir.x);
        animator.SetFloat("flyingDirectionY", Mathf.Abs(localDir.z)); 
    }
}

    /// <summary>
    /// Called by SessionManager after a delay (e.g. 15s) if the patient
    /// still hasn't found this target. Plays a spatialized bird call once
    /// per target -- relies on the AudioSource's 3D spatial blend and the
    /// AR camera's AudioListener for the patient to perceive it as coming
    /// from the neglected side.
    /// </summary>
    public void PlayHintCallIfNeeded()
    {
        if (_hintPlayedForCurrentTarget) return;

        if (audioSource == null)
        {
            Debug.LogWarning($"{name}: PlayHintCallIfNeeded called but no AudioSource is assigned in the Inspector.", this);
            return;
        }
        if (hintCallClips == null || hintCallClips.Length == 0)
        {
            Debug.LogWarning($"{name}: PlayHintCallIfNeeded called but Hint Call Clips is empty in the Inspector.", this);
            return;
        }

        _hintPlayedForCurrentTarget = true;
        AudioClip clip = hintCallClips[Random.Range(0, hintCallClips.Length)];
        audioSource.clip = clip;
        audioSource.Play();
    }

    /// <summary>
    /// Keeps "flying" and "perched" as strict opposites so the Animator
    /// never ends up in an ambiguous state where both or neither are true.
    /// </summary>
    private void SetFlying(bool isFlying)
    {
        if (animator == null) return;
        animator.SetBool("flying", isFlying);
        animator.SetBool("perched", !isFlying);
    }
}