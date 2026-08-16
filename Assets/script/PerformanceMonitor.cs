using UnityEngine;

public class PerformanceMonitor : MonoBehaviour
{
    private float testTime = 30f;
    private float timer = 0f;

    private float totalFrameTime = 0f;
    private float minFrameTime = float.MaxValue;
    private float maxFrameTime = 0f;

    private int frameCount = 0;

    void Update()
    {
        float frameTime = Time.unscaledDeltaTime * 1000f; // milliseconds

        totalFrameTime += frameTime;
        minFrameTime = Mathf.Min(minFrameTime, frameTime);
        maxFrameTime = Mathf.Max(maxFrameTime, frameTime);

        frameCount++;
        timer += Time.unscaledDeltaTime;

        if (timer >= testTime)
        {
            float averageFrameTime = totalFrameTime / frameCount;

            float averageFPS = 1000f / averageFrameTime;

            Debug.Log("===== PERFORMANCE RESULT =====");
            Debug.Log("Frames Recorded = " + frameCount);
            Debug.Log("Average Frame Time = " + averageFrameTime.ToString("F2") + " ms");
            Debug.Log("Average FPS = " + averageFPS.ToString("F2"));
            Debug.Log("Minimum Frame Time = " + minFrameTime.ToString("F2") + " ms");
            Debug.Log("Maximum Frame Time = " + maxFrameTime.ToString("F2") + " ms");

            enabled = false;
        }
    }
}