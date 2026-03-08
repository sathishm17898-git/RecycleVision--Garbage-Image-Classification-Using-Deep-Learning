Developing a machine learning pipeline to classify images of garbage into predefined categories (e.g., plastic, paper, metal, organic, glass).
**Motivation**: Automating waste segregation improves recycling efficiency and supports sustainable waste management programs.
**Approach**: End‑to‑end image classification workflow using deep learning, benchmarked across multiple CNN architectures.
**Dataset**
Source: Custom dataset of garbage images collected and organized into class‑specific folders.
Classes: Plastic, Paper, Metal, Organic, Glass.
**Preprocessing**:
  - Resized images to uniform dimensions (224×224).
  - Normalized pixel values to [0,1].
  - Applied augmentations (rotation, flip, zoom, brightness adjustment) for generalization.
**Splitting Strategy**:
- Train: 70%
- Validation: 15%
- Test: 15%
- Ensured reproducibility with fixed random seeds.

**Model Architectures**
- EfficientNetB0: Balanced accuracy and efficiency.
- MobileNetV2: Lightweight, suitable for deployment.
- ResNet50: High accuracy, deeper architecture.

**Training Pipeline**
  - Frameworks: TensorFlow/Keras
  - Optimizer: Adam with learning rate scheduling.
  - Loss Function: Sparse Categorical Cross‑Entropy.
  - Batch Size: Optimized at 32 for GPU efficiency.
  - Drop out: Monitored validation loss to prevent overfitting.

**Evaluation**
  - Metrics: Accuracy, Precision, Recall, F1‑Score.
  - Visualization:
    - Confusion matrix with labeled axes.

**Deployment**
- Platform: Streamlit app for interactive classification.
- Features:
- Image upload and prediction button.
- Display of predicted class with confidence score.
- Visualization of augmented samples for sanity checks.
- Environment Setup:
- Python 3.12, TensorFlow 2.x

**Key Challenges & Solutions**
- Class Imbalance: Applied class weights.

**Impact & Applications**
- Recruiter‑friendly phrasing:
- “Built and deployed a deep learning pipeline achieving 92% accuracy in automated garbage classification, enabling scalable waste segregation solutions.”
- Real‑world use cases: Smart recycling bins, municipal waste management, sustainability programs.
