using UnityEngine;
using UnityEngine.XR.ARFoundation;

public class ARTrackedImageSpawner : MonoBehaviour
{
    [SerializeField] private GameObject charizardPrefab;
    private GameObject charizard;

    private ARTrackedImageManager arTrackedImageManager;

    private void OnEnable()
    {
        arTrackedImageManager = GetComponent<ARTrackedImageManager>();
        arTrackedImageManager.trackablesChanged.AddListener(OnImageChanged);
    }

    private void OnDisable()
    {
        arTrackedImageManager.trackablesChanged.RemoveListener(OnImageChanged);
    }

    private void OnImageChanged(ARTrackablesChangedEventArgs<ARTrackedImage> eventArgs)
    {
        foreach (var newImage in eventArgs.added)
        {
            charizard = Instantiate(
                charizardPrefab,
                newImage.transform
            );
        }
    }
}