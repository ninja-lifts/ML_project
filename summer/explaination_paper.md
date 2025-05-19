# Understanding and Compressing Vision-Language Models through the Lens of Intrinsic Dimension

## 🚩 Motivation
Vision-Language Models (VLMs) like BLIP and ViLT are overparameterized, but not all parameters contribute equally. This work explores how to identify and remove less important parts of these models using **Intrinsic Dimension (ID)**.

## ✂️ Pruning Using ID
Three pruning strategies are considered:

1. **Magnitude-based**: Remove layers with small weight values.
2. **Gradient-based**: Remove layers with small gradient norms.
3. **PLATON (Proposed)**: Combines Intrinsic Dimension (ID), Utility, and Complexity into a scoring function:

   $$
   \text{Score}_\ell = \frac{\text{ID}_\ell}{\text{Comp}_\ell} + \lambda \cdot \text{Utility}_\ell
   $$

Where:
- **IDₗ**: Intrinsic Dimension of layer \( \ell \), indicating how much new information it adds.
- **Compₗ**: Complexity (e.g., number of parameters) in layer \( \ell \).
- **Utilityₗ**: Drop in performance (e.g., CIDEr) when the layer is removed.
- **λ**: Trade-off hyperparameter.

## 🧠 What is Image Captioning?
Given an image, generate a textual description, e.g.,
> "A dog is playing with a ball on the grass."

It combines:
- **CV** (Computer Vision): understand image content
- **NLP** (Natural Language Processing): generate coherent text

## 📸 What is BLIP?
**BLIP**: Bootstrapped Language-Image Pretraining

BLIP Architecture:
- **Vision Encoder** (e.g., ViT)
- **Text Encoder** (e.g., BERT)
- **Fusion Module** (aligns image & text features)
- **Decoder** (generates captions)

### Example Use
Input: Image of a man skateboarding  
Output: "A man is riding a skateboard down the street."

## 📏 Evaluation Metrics for Captioning

### 🔹 BLEU
- Measures n-gram overlaps
- BLEU-1: unigrams, BLEU-4: 4-grams

### 🔹 METEOR
- Considers synonyms, stems, grammar

### 🔹 ROUGE
- Recall-oriented, measures overlap of relevant words

### 🔹 CIDEr
- Designed for image captioning
- Uses TF-IDF for n-gram weighting
- Compares against multiple human captions

**TF-IDF:**
- **TF**: Word frequency in a caption
- **IDF**: Rarity across captions
- Rare but informative words (e.g., "skateboard") are weighted higher

## 🔍 Intrinsic Dimension (ID)

### Definition
ID = Minimum number of variables needed to describe the data without losing significant information.

### Estimation with TwoNN
1. For each point, compute distances to first (r1) and second (r2) nearest neighbors.
2. Compute \( \mu = r2 / r1 \)
3. Estimate ID by fitting a Pareto distribution (via Maximum Likelihood Estimation).

### ID Patterns in VLMs
- **Visual Layers**: ID ranges from 20–450; shows a 'hunchback' pattern → more sensitive to pruning.
- **Language Layers**: ID ~ 5–30; uniform pattern → robust to pruning.
- **Fusion Layers**: ID between visual & language; periodic → balances complexity.

## 🔧 Pruning Strategies

### 🔹 Traditional Metrics
- **Magnitude-based**: \( S(\theta_i) = |\theta_i| \)
- **Gradient-based**: \( S(\theta_i) = L(\theta) - L(\theta - \theta_i) \)

### 🔹 ID-Enhanced Importance
$$
S(\theta_i) = S(\theta_i) \times \text{ID(layer)}
$$

### 🔁 Iterative Pruning
1. Estimate ID per layer
2. Gradually increase pruning ratio (cubic schedule)
3. Recalculate scores \( S(\theta_i) \times ID \)
4. Fine-tune pruned model

## 🔍 Weight Importance Score
$$
\text{Score}(w) = \mathbb{E}_{x \in D}[\langle \nabla_w \ell(f(x), y), w \rangle]
$$

### Explanation
- **w**: weight  
- **D**: dataset  
- **x, y**: input and label  
- \( \ell(f(x), y) \): loss  
- \( \nabla_w \ell \): gradient w.r.t. weight  
- ⟨⋅,⋅⟩: inner product

**Intuition:**
- Measures how aligned the weight is with the gradient
- Large positive score → weight reduces loss → important
- Small or negative score → weight can be pruned

### Geometric Interpretation
- Alignment of gradient and weight direction
- Similar to Fisher Information in signal theory

## 🔗 Relation to ID
Use weight score **after** ID analysis:
1. Use ID to find redundant (low ID) layers
2. Within those layers, use importance score to prune unhelpful weights

## 🧪 Experimental Results

### Dataset
- **MSCOCO**

### Metrics
- **CIDEr**: Content similarity
- **BLEU@4**: N-gram precision
- **SPICE**: Semantic similarity

### Key Findings
- PLATON outperforms traditional pruning methods
- At 40% pruning, only 1.9 CIDEr drop
- Visual layers are more sensitive (high ID)
- ID-aware pruning leads to smarter compression

## ✅ Conclusion
- ID reveals complexity of representations in different layers
- Combining ID with utility and complexity (PLATON) enables smarter pruning
- Large VLMs can be compressed with minimal performance loss using ID-guided pruning

---

**Takeaway:** Intrinsic Dimension helps identify which layers and weights are crucial, enabling effective model compression without significant drop in task performance.

