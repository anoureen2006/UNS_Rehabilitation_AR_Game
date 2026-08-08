using UnityEngine;

/// <summary>
/// Moves the bird (or star, or any target) GameObject toward a destination
/// at a runtime-configurable speed. Attach to the target prefab itself.
/// SessionManager calls SetTarget() after the target has been spawned and
/// anchored -- this script only handles local movement/animation, it never
/// talks to the network directly.
/// </summary>
public class BirdController : MonoBehaviour
{
    [SerializeField] private float rotationSpeedDeg = 180f;
    [SerializeField] private float arrivalThresholdM = 0.05f;
    [SerializeField] private Animator animator; // optional, leave null if no animation yet

    private Vector3 _targetWorldPos;
    private float _speed = 0.5f;
    private bool _hasTarget = false;

    public bool HasArrived { get; private set; }

    public void SetTarget(Vector3 worldPosition, float speed)
    {
        _targetWorldPos = worldPosition;
        _speed = Mathf.Max(0.05f, speed);
        _hasTarget = true;
        HasArrived = false;

        if (animator != null) animator.SetBool("IsMoving", true);
    }

    public void Stop()
    {
        _hasTarget = false;
        if (animator != null) animator.SetBool("IsMoving", false);
    }

    private void Update()
    {
        if (!_hasTarget || HasArrived) return;

        transform.position = Vector3.MoveTowards(
            transform.position, _targetWorldPos, _speed * Time.deltaTime);

        Vector3 toTarget = _targetWorldPos - transform.position;
        if (toTarget.sqrMagnitude > 0.0001f)
        {
            Quaternion desiredRot = Quaternion.LookRotation(toTarget.normalized, Vector3.up);
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation, desiredRot, rotationSpeedDeg * Time.deltaTime);
        }

        if (Vector3.Distance(transform.position, _targetWorldPos) <= arrivalThresholdM)
        {
            HasArrived = true;
            if (animator != null) animator.SetBool("IsMoving", false);
        }
    }
}
