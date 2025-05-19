# 📘 Abstract Explanation and Notes

## 1. Intrinsic Dimension (ID)

**Original Statement:**
> “The intrinsic dimension (ID) represents the minimum dimension needed to describe data on a lower-dimensional manifold within high-dimensional spaces.”

**🔎 Explanation:**
- Intrinsic Dimension (ID) is a measure of the **true degrees of freedom** in a dataset.
- Even if data exists in a **high-dimensional space**, meaningful patterns often lie on a **lower-dimensional manifold**.
- A manifold is like a curved surface embedded in higher-dimensional space.

**💡 Example:**
- A **spiral on a 2D sheet** exists in 2D, but can be described with just 1 parameter (e.g., angle) → **ID = 1**.
- MNIST digits (784D input) can be embedded into a ~15D manifold.

---

## 2. Network Pruning

**Original Statement:**
> “Network pruning aims to reduce the complexity of high-dimensional networks while minimizing performance trade-offs.”

**🔎 Explanation:**
- Pruning removes **unnecessary weights, neurons, or layers** from a neural network.
- Benefits include:
  - Reduced **memory usage**
  - Faster **inference**
  - Lower **energy consumption**
- Aim: Keep model performance **as close to original as possible**.

**💡 Example:**
- In a CNN trained on CIFAR-10, remove neurons with near-zero weights.

---

## 3. Symmetry Between ID and Pruning

**Original Statement:**
> “This symmetry motivates the exploration of ID as a metric for effective pruning.”

**🔎 Explanation:**
- Both **ID** and **pruning** aim to **remove redundancy**.
- Therefore, ID can potentially guide **which layers are redundant** and **can be pruned**.

---

## 4. Separate Manifolds for Modalities

**Original Statement:**
> “For vision-language models, we investigate whether different modalities exist on separate manifolds, indicating varying complexity and prunability.”

**🔎 Explanation:**
- **Vision-Language Models (VLMs)** handle both **image** and **text** inputs.
- A **modality** = type of input (e.g., vision vs. language).
- Authors ask: Do **vision and language features** lie on **different manifolds**?
- If so, they may have **different ID values**, implying different **prunability**.

**💡 Analogy:**
- Text might lie on a flatter manifold than image features → **easier to prune**.

---

## 5. Studying ID Variations in Large VLMs

**Original Statement:**
> “We empirically study ID variations in large-scale vision-language pre-trained models and examine the contributions of different modalities to model prunability.”

**🔎 Explanation:**
- The authors conduct **experiments** on large pre-trained VLMs (e.g., CLIP, BLIP, Flamingo).
- They analyze:
  - How ID varies across **layers**
  - How **vision and language** components affect **model prunability**

---

## 6. ID-Based Layer Importance Metric

**Original Statement:**
> “We propose a layer importance metric based on ID, which can conveniently integrate with current metrics and enhance performance in vision-language model pruning.”

**🔎 Explanation:**
- A **new pruning metric** is proposed, using **ID to score the importance of layers**.
- This can be combined with traditional metrics like:
  - **Weight magnitude**
  - **Fisher information**
- Helps preserve **critical layers** (high ID), prune **redundant ones** (low ID).

---

## 7. Correlation Between ID and Prunability

**Original Statement:**
> “The experimental results show a high correlation between ID and modality prunability.”

**🔎 Explanation:**
- Experimental results show:
  - **High ID layers** → contain critical information → **hard to prune**
  - **Low ID layers** → contain redundant info → **easy to prune**

---

## 8. Asymmetric Importance of Vision vs. Language

**Original Statement:**
> “Visual representations are more sensitive and crucial to model performance, while language representations are more robust and offer greater prunability.”

**🔎 Explanation:**
- **Visual features**:
  - More complex
  - Higher ID
  - Harder to prune
- **Language features**:
  - Simpler
  - Lower ID
  - Easier to prune with minimal loss
- Shows **asymmetry** in modality sensitivity.

---

## 9. Asymmetric Pruning Strategy

**Original Statement:**
> “Our findings suggest an asymmetric pruning strategy for vision and language modalities, guided by the ID metric.”

**🔎 Explanation:**
- Key proposal:
  - Don’t prune both modalities the same way.
  - Use **ID to guide asymmetric pruning**:
    - Prune **language layers more aggressively**
    - Be **conservative with image layers**

---

