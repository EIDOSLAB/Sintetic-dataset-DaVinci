# Sintetic-dataset-DaVinci

**Synthetic dataset generation and validation for robotic surgery instrument segmentation**

**Repository:** https://github.com/EIDOSLAB/Sintetic-dataset-DaVinci

---

## Overview

This project provides a reproducible pipeline and a photorealistic synthetic dataset of Da Vinci™ surgical instruments for semantic segmentation. The dataset was generated with an automated Python–Autodesk Maya™ pipeline that produces labeled RGB video frames and pixel-accurate binary masks. The synthetic scenes include randomized tool motion, lighting variations, and realistic blood-like textures to emulate intraoperative variability.

The accompanying manuscript ("**Synthetic dataset generation and validation for robotic surgery instrument segmentation**" by Giorgio Chiesa *et al.*) has been **submitted to ISBI 2026**. The submitted paper is included with this repository: `/mnt/data/1571225924 paper.pdf`.

---

## Key features

- Automated pipeline for batch generation of photorealistic synthetic endoscopic sequences.
- Pixel-accurate ground truth masks exported from Maya (object ID rendering).
- Randomized motion patterns, lighting, and texture variations (including blood patches).
- Optional green-screen compositing to overlay instruments onto real surgical backgrounds.
- Reproducible dataset generation using seeded randomness for deterministic runs.
- Validation experiments showing that mixing synthetic with real images improves segmentation generalization.

---

## Dataset contents

Typical repository layout:

```
/data
  /synthetic
    /images       # RGB frames (1920x1080)
    /masks        # Binary segmentation masks (same resolution)
  /real
    /images       # Real frames used for validation
    /masks        # Manually annotated masks (Label Studio)
README.md
paper.pdf         # Submitted to ISBI 2026
scripts/          # Maya + Python scripts to generate the dataset
models/           # Example training scripts and checkpoints
```

> Note: The exact folder names and structure in your clone may vary. Check the repository root for the definitive layout.

---

## Quick start — use the dataset

1. Clone the repository:
```bash
git clone https://github.com/EIDOSLAB/Sintetic-dataset-DaVinci.git
cd Sintetic-dataset-DaVinci
```

2. Inspect data (example):
```bash
ls data/synthetic/images | head
ls data/synthetic/masks  | head
```

3. Example training (PyTorch / U-Net):
- Preprocess images to the chosen input size (e.g. 256×256).
- Use standard augmentations (flips, translations, color jitter).
- Train with a combined real/synthetic ratio `α` (see paper): experiments show balanced mixes (e.g. α ≈ 0.4–0.5) often improve generalization.

See `scripts/` for provided training examples and reproduction instructions.

---

## Reproducibility & generation

The generation pipeline is implemented in Python and runs in Autodesk Maya's standalone mode. Key script capabilities:
- Import and clean photogrammetry-derived meshes of the Da Vinci arms.
- Procedural texture placement including blood patches.
- Randomized animation of multiple arms using seeded sinusoidal curves.
- Automatic rendering of RGB frames and per-object ID masks.
- Greenscreen generation and automatic playlist-based batch rendering.

To reproduce the synthetic dataset you will need:
- Autodesk Maya with Python API access (used in batch/standalone mode).
- The `scripts/` folder from this repository (Maya scripts and helper utilities).
- The 3D models (photogrammetric meshes) present in `models/` or a path configured in the generator scripts.

---

## Validation & results (summary)

The submitted paper evaluates segmentation performance for U-Net backbones (ResNet18, VGG16, ResNeXt-50) trained with varying fractions of synthetic images (`α`). Main findings:
- A **balanced mix** of real and synthetic images improves IoU on real test sets compared to using only real images.
- Excessive reliance on synthetic-only training can introduce domain shift that degrades performance.
- With a small-scale real set, adding ~40–50% synthetic images substantially increased IoU (reported in the manuscript).

For full experimental details, figures, and quantitative results, see the submitted manuscript included in this repository.

---

## Citation

If you use this dataset or the generation pipeline, please cite the submitted work:

G. Chiesa, R. Borra, V. Lauro, S. De Cillis, D. Amparore, C. Fiori, R. Renzulli, M. Grangetto,  
**"Synthetic dataset generation and validation for robotic surgery instrument segmentation,"** submitted to *ISBI 2026*.  
Paper (submitted): `/mnt/data/1571225924 paper.pdf`

### Suggested BibTeX
```bibtex
@unpublished{Chiesa2025_SinteticDaVinci,
  title = {Synthetic dataset generation and validation for robotic surgery instrument segmentation},
  author = {G. Chiesa and R. Borra and V. Lauro and S. De Cillis and D. Amparore and C. Fiori and R. Renzulli and M. Grangetto},
  note = {Submitted to ISBI 2026. Paper available in repository},
  year = {2025}
}
```

---

## License

See the `LICENSE` file in the repository. If no license is included, please contact the authors before reusing the data or code.

---

## Contact & contributions

If you have questions, want to reproduce the experiments, or contribute:
- Open an issue or pull request on the GitHub repository: https://github.com/EIDOSLAB/Sintetic-dataset-DaVinci
- For direct contact, please use the corresponding author's details as listed in the manuscript.

---

## Acknowledgements

This work was developed at the University of Turin. The project makes use of Autodesk Maya, photogrammetry tools, and open-source deep learning frameworks. Ethical compliance: the study was performed in accordance with the Declaration of Helsinki and approved by the institutional ethics committee; all patient data were anonymized.

---

**Status:** Manuscript submitted to ISBI 2026 — see included `paper.pdf`.
