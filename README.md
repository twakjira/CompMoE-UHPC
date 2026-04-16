<div align="center">

# CompMoE-CFI

### Compositional Mixture-of-Experts Neural Network for Compressive Strength Prediction and Sustainable Mix Optimization of Fiber-Reinforced Ultra-High Performance Concrete

<p>
<a href="#">
<img alt="Paper" src="https://img.shields.io/badge/Paper-Under%20Review-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white">
</a>
<a href="#">
<img alt="Project Page" src="https://img.shields.io/badge/Project-Page-4A90D9?style=for-the-badge&logo=github-pages&logoColor=white">
</a>
<a href="#">
<img alt="Code" src="https://img.shields.io/badge/Code-On%20Acceptance-gray?style=for-the-badge&logo=github&logoColor=white">
</a>
<a href="https://www.sciencedirect.com/science/article/pii/S235234092500900X">
<img alt="Dataset" src="https://img.shields.io/badge/Dataset-Mendeley-green?style=for-the-badge&logo=elsevier&logoColor=white">
</a>
</p>

**[Tadesse G. Wakjira](https://ai4riselab.com)<sup>1</sup>**

<sup>1</sup>Kennesaw State University

*Under Review*

</div>

---

## Overview

**CompMoE-CFI** is a compositional mixture-of-experts neural network for predicting the 28-day compressive strength of fiber-reinforced ultra-high performance concrete (UHPC) and optimizing mix designs for environmental sustainability.

The model decomposes 31 engineered UHPC features into **five physically meaningful subsystems** (Binder, Fiber, Aggregate, Environment, Testing) and introduces three novel mechanisms:

| Mechanism | Description |
|-----------|-------------|
| **CAG** (Compositional Attention Gating) | Multi-head cross-attention over subsystem embeddings produces soft expert routing weights |
| **PIRP** (Physics-Informed Residual Path) | Learnable linear skip connection (α ≈ 0.57) separates near-linear Abrams-law behavior from nonlinear interactions |
| **HED** (Hierarchical Expert Diversity) | 5 heterogeneous experts (1 linear + 2 shallow + 2 residual) matched to compositional complexity |

## Status

The manuscript is currently **under review**. Full source code, training scripts, and trained weights will be released in this repository upon paper acceptance.

## Citation

```bibtex
@article{wakjira2026compmoe,
    title   = {Compositional Mixture-of-Experts Neural Network for Compressive
               Strength Prediction and Sustainable Mix Optimization of
               Fiber-Reinforced Ultra-High Performance Concrete},
    author  = {Wakjira, Tadesse G.},
    year    = {2026},
    note    = {Under Review}
}
```

---

Developed by [AI4RISE Lab](https://ai4riselab.com).
