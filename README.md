# Bone Segmentation — PyTorch Project

This repository contains a small PyTorch pipeline for binary bone segmentation on X-ray images.
It supports:

- Training + validation (with Dice score)
- Automatic checkpoint saving
- Test-time inference without ground-truth masks
- Saving predicted masks
- Simple learning-curve visualization

The default model is a UNet-like architecture
