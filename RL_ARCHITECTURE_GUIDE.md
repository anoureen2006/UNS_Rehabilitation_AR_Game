Here is your complete, updated research-grade document incorporating the **4 Dynamic RL Actions (`speed`, `eccentricity_deg`, `distance_m`, `time_limit_s`)**, the **Fixed `target_count` = 3**, the **Left/Right Neglect support**, and the **Two-Phase Pre-training & Continuous Adaptive Architecture**:

---

# AR Rehabilitation RL Controller Architecture

```mermaid
graph TD
    subgraph Phase1 [Phase 1: Pre-training Base Model (Offline)]
        SimEnv[UnityARRehabEnv Simulation] --> TrainBase[Train Base Model: train_unity.py]
        TrainBase --> BaseCheckpt[(base_ppo_model.zip)]
    end

    subgraph Phase2 [Phase 2: Continuous Adaptive Training in Unity (Live Session)]
        BaseCheckpt --> LoadPatient[Load Model for Patient: demo01]
        LoadPatient --> PlayTrial[Patient Plays AR Trial in Unity]
        PlayTrial --> SendTelemetry[Send Telemetry JSON to Server]
        SendTelemetry --> StateVec[15-Dim Observation Vector s_t]
        StateVec --> PolicyNet[PPO Neural Network π_θ]
        PolicyNet --> ActionDec[Select 4 Actions: speed, eccentricity, distance, time_limit]
        ActionDec --> ComputeReward[Calculate Live Reward R_t]
        ComputeReward --> FineTune[Continuous Adaptation: model.learn reset_num_timesteps=False]
        FineTune --> SavePatient[(patient_demo01_ppo.zip)]
        SavePatient --> ActionDec
    end
```

---

## 1. What are the Components in the RL System?

The Reinforcement Learning system operates as a Markov Decision Process (MDP) defined by 5 core components:

### 1. Environment ($\mathcal{E}$) — `UnityARRehabEnv`
The mathematical environment that receives target difficulty settings, simulates patient cognitive-perceptual responses (gaze, reaction time, neglect attenuation, fatigue), and updates session statistics.

### 2. State / Observation Vector ($\mathbf{s}_t \in \mathbb{R}^{15}$)
A **15-dimensional normalized vector** extracted directly from your Unity session trial logs:

| Index | Feature | Range | Description |
| :---: | :--- | :---: | :--- |
| `obs[0]` | `last_hit` | $0.0 \text{ or } 1.0$ | $1.0$ if previous trial was a hit, $0.0$ if miss/timeout. |
| `obs[1]` | `last_rt_norm` | $[0.0, 1.0]$ | Normalized reaction time ($\text{RT} / 30,000\text{ms}$). |
| `obs[2]` | `last_gaze_offset` | $[0.0, 1.0]$ | Patient head/gaze offset angle ($\text{gaze\_deg} / 90^\circ$). |
| `obs[3]` | `is_neglected` | $0.0 \text{ or } 1.0$ | $1.0$ if target was in neglected hemifield. |
| `obs[4]` | `current_speed` | $[0.0, 1.0]$ | Target movement speed ($0.2 - 0.8\text{ m/s}$). |
| `obs[5]` | `current_eccentricity`| $[0.0, 1.0]$ | Target angle into neglected field ($5^\circ - 35^\circ$). |
| `obs[6]` | `current_distance` | $[0.0, 1.0]$ | Depth distance of target ($0.8 - 2.5\text{m}$). |
| `obs[7]` | `target_count` | $[0.0, 1.0]$ | Number of visual targets ($\text{count} / 5.0$, fixed at 3). |
| `obs[8]` | `time_limit` | $[0.0, 1.0]$ | Trial time limit ($\text{time\_limit\_s} / 45.0\text{s}$). |
| `obs[9]` | `rolling_hit_rate` | $[0.0, 1.0]$ | Average success rate over last 5 trials. |
| `obs[10]`| `rolling_avg_rt` | $[0.0, 1.0]$ | Average reaction time over last 5 trials. |
| `obs[11]`| `consecutive_timeouts`| $[0.0, 1.0]$ | Count of back-to-back timeout failures. |
| `obs[12]`| `neglect_side_is_left` | $0.0 \text{ or } 1.0$ | $1.0$ for Left Spatial Neglect, $0.0$ for Right Neglect. |
| `obs[13]`| `session_progress` | $[0.0, 1.0]$ | Trial progress ratio ($\text{trial\_idx} / 60$). |
| `obs[14]`| `fatigue_estimate` | $[0.0, 1.0]$ | Accumulated cognitive fatigue factor. |

### 3. Action Vector ($\mathbf{a}_t \in \mathcal{A}$)
The RL policy selects **4 dynamic difficulty parameters** for the next Unity trial:

| Action # | Action Parameter | Range | Description |
| :---: | :--- | :---: | :--- |
| **1** | **`speed`** | `0.2 - 0.8 m/s` | Target movement speed. |
| **2** | **`eccentricity_deg`** | `5.0° - 35.0°` | Angular placement into the neglected hemifield (Left/Right). |
| **3** | **`distance_m`** | `0.8 - 2.5 m` | Target depth in 3D AR space. |
| **4** | **`time_limit_s`** | `10.0 - 45.0 s` | Dynamic trial timeout window before timeout occurs. |
| *Fixed* | *`target_count`* | *`3`* | *Fixed constant for visual clutter control.* |

### 4. Reward Function ($R_t$) — `compute_unity_reward`
The mathematical incentive signal guiding the RL policy:
$$R_t = \begin{cases} 
+ 15.0 \cdot \left(1 + \frac{\text{eccentricity}}{15^\circ}\right) - \frac{\text{RT} - 4000}{1000} & \text{if Hit in Neglected Hemifield} \\
+ 3.0 & \text{if Hit in Non-Neglected Hemifield} \\
- 15.0 & \text{if Timeout Failure ($\text{RT} \ge \text{time\_limit\_s}$)}
\end{cases}$$

### 5. Policy Network ($\pi_\theta$)
Multi-Layer Perceptron (MLP) neural network mapping observation vectors $\mathbf{s}_t \to \mathbf{a}_t$.

---

## 2. The Two-Phase Training & Adaptation Strategy

```
  Phase 1: Pre-training Base Model        Phase 2: Continuous Live Adaptation
     (Offline Simulation)                      (Real-Time Unity Session)
 ┌───────────────────────────┐             ┌─────────────────────────────┐
 │ • Pre-trains on simulated │             │ • Loads base model for      │
 │   patients over 50,000    │  ───────►   │   new patient demo01        │
 │   trials.                 │             │ • Fine-tunes weights live   │
 │ • Output:                 │             │   after every trial!        │
 │   base_ppo_model.zip      │             │ • Output:                   │
 └───────────────────────────┘             │   patient_demo01_ppo.zip    │
                                           └─────────────────────────────┘
```

### Phase 1: Pre-training the Base Model (Offline Simulation)
* **Goal**: Establish a safe baseline policy before any real patient uses the AR headset.
* **Execution**: Run `python train_unity.py`.
* **Output**: `checkpoints/base_ppo_model.zip` (Ensures safe initial defaults such as $10^\circ\text{ eccentricity, } 0.3\text{ m/s speed}$).

### Phase 2: Continuous Online Adaptive Fine-Tuning (Live Unity Session)
* **Goal**: Personalize the neural network in real-time to patient `demo01`'s specific neglect border, motor speed, and fatigue rate.
* **Execution**: As `demo01` completes trials in Unity AR, the Python server executes online fine-tuning steps using Stable-Baselines3:
  ```python
  # Fine-tune model weights continuously on live patient trial feedback
  model.learn(total_timesteps=5, reset_num_timesteps=False)
  # Save patient-specific checkpoint
  model.save("checkpoints/patients/demo01_ppo.zip")
  ```

---

## 3. How Can We Use It in Unity AR Foundation?

### REST API Server (`unity_continuous_api.py`)

Unity sends trial telemetry to Python after each trial, and Python responds with the 4 action parameters in $< 10\text{ms}$:

#### Unity C# Script (`ARFoundationRLController.cs`):
```csharp
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.XR.ARFoundation;
using Newtonsoft.Json;

public class ARFoundationRLController : MonoBehaviour
{
    public Transform arCameraTransform; 
    public GameObject targetPrefab;     
    public string serverUrl = "http://localhost:5000/predict_and_adapt";

    public void OnTrialComplete(string sessionJson)
    {
        StartCoroutine(SendRLRequest(sessionJson));
    }

    IEnumerator SendRLRequest(string jsonPayload)
    {
        UnityWebRequest request = new UnityWebRequest(serverUrl, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonResponse = request.downloadHandler.text;
            var responseData = JsonConvert.DeserializeObject<RLResponse>(jsonResponse);
            var diff = responseData.recommended_next_trial.difficulty_at_trial;

            // Spawn target at recommended eccentricity, distance, speed, and time_limit_s
            SpawnTargetInAR(diff.eccentricity_deg, diff.distance_m, diff.speed, diff.time_limit_s);
        }
    }

    void SpawnTargetInAR(float eccentricityDeg, float distanceMeters, float speed, float timeLimitSeconds)
    {
        Vector3 cameraPos = arCameraTransform.position;
        Vector3 cameraForward = arCameraTransform.forward;

        // Apply angular offset into neglected hemifield (negative angle for left)
        Quaternion rotation = Quaternion.AngleAxis(-eccentricityDeg, Vector3.up);
        Vector3 spawnDirection = rotation * cameraForward;
        Vector3 spawnPosition = cameraPos + (spawnDirection * distanceMeters);

        GameObject targetInstance = Instantiate(targetPrefab, spawnPosition, Quaternion.identity);
        targetInstance.transform.LookAt(arCameraTransform);
        
        var movementScript = targetInstance.GetComponent<TargetMovement>();
        if (movementScript != null) {
            movementScript.moveSpeed = speed;
            movementScript.timeLimit = timeLimitSeconds;
        }
    }
}

[System.Serializable]
public class RLResponse {
    public string session_id;
    public string patient_id;
    public RecommendedTrial recommended_next_trial;
}

[System.Serializable]
public class RecommendedTrial {
    public DifficultySettings difficulty_at_trial;
}

[System.Serializable]
public class DifficultySettings {
    public float speed;
    public float eccentricity_deg;
    public float distance_m;
    public int target_count;
    public float time_limit_s;
}
```

---

## 4. What is the RL Technique That We Use?

We implement and compare **3 primary Deep Reinforcement Learning algorithms**:

```
                       Deep RL Techniques Used
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
   PPO (Recommended)             DQN                      A2C
(Proximal Policy Optim.)   (Deep Q-Network)     (Advantage Actor-Critic)
 Policy-Gradient Method    Value-Based Method     Synchronous Actor-Critic
```

### 1. PPO (Proximal Policy Optimization) — *Primary Recommended Technique*
* **Type**: On-policy, Actor-Critic Policy Gradient.
* **Why PPO**: Uses a **clipped surrogate objective function**:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
  This prevents destructively large policy updates, ensuring smooth, stable difficulty adaptation during live patient sessions.

### 2. DQN (Deep Q-Network)
* **Type**: Off-policy, Value-based Q-Learning.
* **Why DQN**: Learns action-value functions $Q(s, a)$ using experience replay buffers to evaluate discrete difficulty steps.

### 3. A2C (Advantage Actor-Critic)
* **Type**: Synchronous Actor-Critic.
* **Why A2C**: Combines policy network updates with an estimated baseline value function $V(s)$ to reduce variance.

---

## 5. What is the Objective in the RL?

The overall objective of the RL agent is to act as an **Intelligent Automated Physical Therapist** that maximizes **Rehabilitation Quality & Spatial Neglect Recovery**:

1. **Scaffolding the Neglected Border (Vygotsky Zone of Proximal Development)**:
   - Rather than making harsh jumps ($10^\circ \to 20^\circ$) that cause $30\text{s}$ timeouts, the RL policy learns to **gradually expand** the patient's neglected scanning boundary ($10^\circ \to 12.5^\circ \to 14^\circ \to 16^\circ$).

2. **Dynamic Trial Timeout Optimization**:
   - Dynamically adjust `time_limit_s` ($10\text{s} - 45\text{s}$) to match patient reaction speed, eliminating disengagement and frustration.

3. **Dynamic Fatigue & Attention Management**:
   - Automatically detect when reaction times begin to degrade due to fatigue, temporarily easing off target speed/distance to allow recovery before challenging the patient again.
