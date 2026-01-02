# Dynamic DPO Training Walkthrough

## Overview

This document explains the training logic of **Dynamic DPO (Direct Preference Optimization)** located in `archive/dynamic_beta`.

**Core Concept:** Unlike standard DPO, which uses a fixed accumulation coefficient (`beta`) throughout training, Dynamic DPO treats `beta` as an adaptive hyperparameter. It continuously adjusts `beta` based on the model's "confidence" (margins) to keep training stable and prevent over-optimization (reward hacking).

## Key Differences: Dynamic Beta vs. Original DPO

| Feature | Original DPO | Dynamic DPO |
| :--- | :--- | :--- |
| **Beta Parameter** | Fixed (e.g., `beta=0.1`). | Adaptive (changes every step). |
| **Constraint Mechanism** | Constant penalty for deviating from the reference. | Strength of penalty increases if model gets too confident/aggressive. |
| **Training Phases** | Single phase (Standard Training). | Two phases: **Warmup** (learn baseline) $\rightarrow$ **Dynamic** (adaptive adjustment). |
| **Risk Management** | None (relies on manual tuning). | **Risk Test**: Statistical check to detect if model is "drifting" too fast. |

---

## Logic Walkthrough

### 1. The Warmup Phase

**Goal:** Establish a baseline for "normal" model behavior before we start adjusting anything.

- **Duration:** Defined by `warmup_steps` (e.g., first 100 steps).
- **Behavior:**
  - The model trains exactly like standard DPO with a fixed `beta_0`.
  - **Crucial Step:** It collects the **Margins** ($log \pi_{chosen} - log \pi_{rejected}$) for every batch.
  - These margins are fed into a `WarmupQuantileAccumulator`.
- **Outcome:** At the end of the warmup, we calculate an initial threshold $\tau_0$ (e.g., the 90th percentile of margins). This represents the "expected top confidence" of the model.

### 2. The Dynamic Evaluation Loop

After warmup, for every single training step, the following logic executes:

#### A. Calculate Margins

For the current batch of data, we calculate the **Model Margin ($M$)**:
$$ M = ( \log \pi(y_w|x) - \log \pi(y_l|x) ) - ( \log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x) ) $$
*Ideally, we want this to be positive (model prefers winner more than reference does).*

#### B. Update Threshold ($\tau$)

We maintain a moving average of the margin threshold using exponential decay:
$$ \tau_{new} = (1 - \lambda) \cdot \tau_{old} + \lambda \cdot \text{BatchQuantile}(M) $$
*This ensures our standard for "over-confidence" evolves as the model learns.*

#### C. The Risk Test

We check if the model is producing too many "high-margin" samples (outliers).

1. **Count Outliers ($p_{hat}$)**: Proportion of samples in the current batch where $Margin > \tau$.
2. **Statistical Test**: Is $p_{hat}$ significantly higher than our target rate ($\delta$)?
   - Significance is determined by an upper bound $\epsilon$ (based on batch size).
   - If $p_{hat} > \delta + \epsilon$, the **Risk Test Fails** (Result: `is_over_risk = True`).

#### D. Adjust Beta

Based on the Risk Test, we update `beta`:
$$ u_k = \frac{p_{hat} - (\delta + \epsilon)}{\epsilon} $$
$$ s_k = \tanh(\gamma \cdot u_k) $$
$$ \beta_{new} = \beta_{old} \cdot \exp(\alpha \cdot s_k) $$

- **If Risk is High ($s_k > 0$):** `beta` **INCREASES**.
  - **Effect:** Stronger penalty for deviation. Forces the model to stick closer to the reference. "Pulls the leash tighter."
- **If Risk is Low ($s_k < 0$):** `beta` **DECREASES**.
  - **Effect:** Weaker penalty. Allows the model to explore and be more aggressive. "Loosens the leash."

### 3. Compute Loss

The final DPO loss is computed using this *newly adjusted* `beta`:
$$ Loss = - \log \sigma( \beta_{new} \cdot M ) $$

---

## Code Map

- **`WarmupQuantileAccumulator`**: Collects stats during Phase 1.
- **`EMAUpdate`**: Updates $\tau$ (the moving goalpost) in Phase 2.
- **`empirical_over_threshold_proportion`**: Calculates $p_{hat}$ (current outlier rate).
- **`risk_test`**: Boolean check for `is_over_risk`.
- **`update_beta`**: The math formula that actually changes the `beta` variable.

## Summary

Dynamic DPO is a **self-stabilizing** algorithm. Instead of manually guessing the perfect `beta`, you give the model a "target outlier rate" ($\delta$). If the model produces margins that are too extreme (over-optimization), the algorithm automatically brakes (increases beta). If the model is too conservative, it accelerates (decreases beta).
