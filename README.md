# PIP-Bench

## 🚀 Research Overview: Standardizing Evaluation for Personalized Image Generation

While personalized image generation has seen rapid advancement, the field still lacks a standardized and reliable evaluation framework for its two core axes: **Identity (ID) Preservation** and **Text Alignment.** Current metrics often fail to capture the nuances of human-centric generation. To bridge this gap, our research identifies the limitations of existing benchmarks and proposes a more robust, explainable evaluation protocol.

## ✨ Key Contributions
### 1. 🔍 Systematic Analysis of Existing Metrics
We conduct a deep dive into widely used ID and text-alignment metrics, identifying critical failure modes in personalized human image generation:

- Factor Sensitivity: Vulnerability to changes in pose, lighting, or background.

- Semantic/Demographic Grouping: Group identities based on semantic/demographic biases rather than discriminate identities based on facial features. 

- Lack of Explainability: Quantitative scores that fail to provide "why" a model failed.

### 2. 🧠 Rubric-Guided MLLM Evaluator
We introduce an MLLM-based Evaluator designed for transparent and diagnostic assessment:

- Normalized Scoring: Outputs standardized ID and text-alignment scores guided by a structured rubric to ensure consistency.

- Explainable Failure Analysis: Beyond simple scores, it utilizes Multiple Choice Questions (MCQ) to diagnose specific reasons for alignment or identity loss.

### 3. 📊 PIP-Bench: A New Standard Benchmark
To unify evaluation efforts, we present PIP-Bench (Personalized Image Perception Benchmark):

- Curated Identities: A high-quality set of diverse identities for rigorous testing.

- Structured Multi-Factor Prompts: Includes a hierarchy of Single, Double, and Triple-factor prompts to evaluate how models handle increasing prompt complexity.

= Standardized Protocol: A reproducible evaluation pipeline to ensure fair comparison across different generative models.

You can access the [images](./PIP-Bench) and [prompts](./prompts.json) for PIP-Bench. 
