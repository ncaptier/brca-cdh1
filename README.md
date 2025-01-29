# brca-cdh1

This repository proposes a Pytorch implementation of a transformer model with a distance-aware self-attention mechanism to predict the CDH1 mutational status of breast cancer patients from Whole Slide Images (WSI).

This Python implementation is inspired by the [STAMP protocol](https://github.com/KatherLab/STAMP) as well as by [a work from Ali Haider Ahmad](https://github.com/AliHaiderAhmad001/Self-Attention-with-Relative-Position-Representations).

## Data & Pre-processing
Diagnostic slides and clinical data (i.e., CDH1 status) from TCGA-BRCA are available via the [GDC portal](https://portal.gdc.cancer.gov/projects/TCGA-BRCA).

WSIs were pre-processed following the [STAMP protocol](https://github.com/KatherLab/STAMP):

* WSIs were split into 224x224 pixel patches and background patches were removed.
* Embedding vectors were extracted for each patch, using the [UNI feature extractor](https://huggingface.co/MahmoodLab/UNI).
* The coordinates of each patch were saved (useful to compute the pairwise euclidean distances).

## Acknowledgements

This project was carried out by [Nicolas Captier](https://ncaptier.github.io/) and supervised by [Georg Wölflein](https://georg.woelflein.eu/).

## References

* [[1]](https://doi.org/10.1038/s41596-024-01047-2) El Nahhas et al. From whole-slide image to biomarker prediction: end-to-end weakly supervised deep learning in computational pathology. Nat Protoc (2025)
* [[2]](https://arxiv.org/abs/1803.02155) Shaw et al. Self-Attention with Relative Position Representations. arXiv (2018)
* [[3]](https://arxiv.org/abs/2305.10552) Wölflein et al. Deep Multiple Instance Learning with Distance-Aware Self-Attention. arXiv (2023)
