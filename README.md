# AR Rehabilitation RL Simulation (UNS / Spatial Neglect Digital Pet Placement)

This repository module contains a complete Reinforcement Learning (RL) simulation for an **Augmented Reality (AR) rehabilitation system** designed for stroke patients with **Unilateral Spatial Neglect (USN)**.

---

## 💡 Main Concept

The AR rehabilitation room contains **ONE virtual digital pet**.
The patient's mission is simply: *"Explore the room and find the digital pet."*

> **Key Architecture**: The RL agent **does not control the patient**. Instead, the RL agent dynamically decides **WHERE the digital pet should appear** in the 10m x 10m room ($20 \times 20$ grid = 400 cells) to maximize:
> - Room exploration
> - Neglected-side (left) exploration
> - Rehabilitation effectiveness
> - Patient engagement

---

## 📂 Codebase Structure

| File | Description |
| :--- | :--- |
| `config.py` | Configuration file for room size (10m x 10m), grid size (20x20), USN severity, patient default parameters, reward weights, and training settings. |
| `patient.py` | Mathematical model simulating a stroke patient with USN (left spatial neglect), FOV cone ($120^\circ$), walking speed, fatigue, working memory, obstacle repulsion, and rightward turning bias. |
| `digital_pet.py` | Virtual AR Digital Pet object representation, cell ID mapping (0..399), continuous coordinate transforms, and reach detection. |
| `reward.py` | Rehabilitation reward shaping function penalizing premature pet discovery and rightward bias while rewarding deep left-side exploration and coverage balance. |
| `environment.py` | Gymnasium environment (`ARRehabEnv`) with 44-dimensional normalized observation vector and 400-discrete action space. |
| `utils.py` | Seed management, evaluation metric aggregation, and baseline heuristic policies (Random, Center, Extreme Left). |
| `visualization.py` | Matplotlib 2D room layouts, 20x20 exploration heatmaps, SB3 learning curves, and Plotly interactive 3D surface/trajectory HTML charts. |
| `train.py` | Training script for **PPO**, **DQN**, and **A2C** algorithms using Stable-Baselines3 with custom callback logging. |
| `evaluate.py` | Benchmarking script comparing baseline heuristics vs trained RL models across 50 test episodes per policy. |
| `app.py` | **Interactive Streamlit Web Application** for live simulation, parameter tuning, algorithm comparison, and Unity AR telemetry JSON export. |
| `run_all.py` | Master pipeline execution script running training, evaluation, and plot generation. |
| `rehab_ar_simulation.ipynb` | Kaggle / Google Colab ready executable Jupyter notebook. |
| `test_simulation.py` | PyUnittest suite verifying environment step mechanics, patient USN turning dynamics, grid cell mapping, and reward signals. |

---

## 🚀 How to Run

### 1. Run Unit Tests
```bash
python test_simulation.py
```

### 2. Launch Interactive Streamlit Web Application
```bash
python -m streamlit run app.py
```

### 3. Run Full RL Training & Benchmark Evaluation
```bash
python run_all.py
```

---

## 📲 Unity AR Integration & Telemetry Export

The Streamlit web application (`app.py`) includes a dedicated export tab that outputs target coordinates $(x, y, z)$ and rehabilitation metrics in standard JSON format ready for consumption by Unity AR headsets (Meta Quest, HoloLens, iOS ARKit):

```json
{
  "sessionId": "REHAB_USN_42",
  "patientProperties": {
    "neglectSeverity": 0.7,
    "walkingSpeed": 0.8,
    "fovDegrees": 120.0,
    "fatigue": 0.05
  },
  "digitalPetSpawnTarget": {
    "cellId": 182,
    "unityPosition": {
      "x": 1.75,
      "y": 0.0,
      "z": 4.25
    }
  },
  "sessionMetrics": {
    "reward": 28.5,
    "leftExplorationPercentage": 45.2,
    "totalExplorationPercentage": 62.0,
    "timeToPetSeconds": 24.5,
    "rehabilitationSuccess": true
  }
}
```
