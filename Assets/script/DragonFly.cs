using UnityEngine;

public class DragonFly : MonoBehaviour
{
    [Header("Targets")]
    public Transform[] targets;

    [Header("Movement")]
    public float moveSpeed = 1.2f;
    public float rotationSpeed = 3f;
    public float stopDistance = 0.5f;

    private Transform currentTarget;
    private bool reachedTarget = false;

    void Start()
    {
        if (targets.Length > 0)
        {
            int randomIndex = Random.Range(0, targets.Length);
            currentTarget = targets[randomIndex];
        }
    }

    void Update()
    {
        if (currentTarget == null || reachedTarget)
            return;

        Vector3 direction = currentTarget.position - transform.position;

        // دوران تدريجي
        if (direction != Vector3.zero)
        {
            Quaternion lookRotation = Quaternion.LookRotation(direction);
            transform.rotation = Quaternion.Slerp(
                transform.rotation,
                lookRotation,
                rotationSpeed * Time.deltaTime
            );
        }

        // الحركة
        if (direction.magnitude > stopDistance)
        {
            transform.position = Vector3.MoveTowards(
                transform.position,
                currentTarget.position,
                moveSpeed * Time.deltaTime
            );
        }
        else
        {
            reachedTarget = true;
            Debug.Log("وصل إلى الهدف");
        }
    }
}