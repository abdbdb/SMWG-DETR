# SMWG-DETR

<p align="center">
  <b>DETR Enhanced by Fourier Spectral Modulation and Wavelet-Guided Fusion</b><br>
  <i>for Tiny Object Detection</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Tiny%20Object%20Detection-blue">
  <img src="https://img.shields.io/badge/Framework-DETR-orange">
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-lightgrey">
</p>

---

This repository provides the **official implementation** of the method proposed in the paper:

> **SMWG-DETR: DETR Enhanced by Fourier Spectral Modulation and Wavelet-Guided Fusion  
> for Tiny Object Detection**

---

## 🔍 Overview

Tiny object detection is a crucial task in the intelligent interpretation of remote sensing imagery, with significant applications in transportation, public security, and emergency management. However, the performance of existing detectors in remote sensing scenarios is still constrained by the extremely small object sizes and the presence of complex background clutter.
In this paper, we propose DETR enhanced by Fourier Spectral Modulation and Wavelet-Guided Fusion (SMWG-DETR), which addresses the issues of spectral distribution bias during feature extraction as well as feature misalignment and detailed feature loss during feature fusion. First, Fourier spectral modulation is employed to suppress redundant frequency components in single-scale feature maps while preserving critical ones, thereby reducing spurious responses caused by cluttered backgrounds. Second, in the feature fusion stage, we apply Discrete Wavelet Transform (DWT) to lower-level feature maps, where the resulting low-frequency and high-frequency sub-bands are used to guide higher-level feature map upsampling and detailed feature refinement, thus leveraging the complementary information across multi-scale features. Finally, a Dynamic Denoising Query Selection (DDQS) strategy is introduced to discard potentially misleading queries in the contrastive denoising process, providing more accurate supervision during training.
In experiments conducted on the AI-TOD and AI-TODv2 datasets, SMWG-DETR achieves average precision scores of **32.1%** and **30.5%**, respectively, achieving state-of-the-art performance.

---

## 📦 Code Availability

🚧 The complete source code, including:

- training code  
- evaluation code  
- model configuration files  

is currently **under preparation** and will be **released publicly upon acceptance of the paper**.

The codebase is being carefully organized and documented to facilitate future use by the research community.

---

## 📌 Status

- ✅ Repository initialized  
- ⏳ Code release (in preparation)  

---

## 📢 Updates

The repository will be updated once the paper is accepted.  
⭐ Please stay tuned for future releases.
