# MLA3DVG: Dual-Modality Multi-level Semantic Alignment for Robust Monocular 3D Visual Grounding

<p align="center">
  <a href="https://arxiv.org/abs/2411.16801"><b>Kevin Qu</b></a><sup>1,2*</sup>, 
  <a href="https://hzqi.github.io/"><b>Haozhe Qi</b></a><sup>3</sup>, 
  <a href="https://mihaidusmanu.com/"><b>Mihai Dusmanu</b></a><sup>1</sup>, 
  <a href="https://mahdirad.github.io/"><b>Mahdi Rad</b></a><sup>1</sup>, 
  <a href="https://ruiwang-pku.github.io/"><b>Rui Wang</b></a><sup>1</sup>, 
  <a href="https://people.inf.ethz.ch/pomarc/"><b>Marc Pollefeys</b></a><sup>1,2</sup>
</p>

<p align="center">
  <sup>1</sup><b>Microsoft Spatial AI Lab</b> &nbsp;&nbsp; <sup>2</sup><b>ETH Zurich</b> &nbsp;&nbsp; <sup>3</sup><b>EPFL</b>
</p>

<p align="center">
  <i>*work done during an internship at Microsoft</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2411.16801"><img src="https://img.shields.io/badge/arXiv-2411.16801-B31B1B.svg" alt="arXiv"></a>
  <a href="https://microsoft.github.io/Loc3R-VLM/"><img src="https://img.shields.io/badge/Project%20Page-Loc3R--VLM-blue.svg" alt="Project Page"></a>
</p>

---

We introduce **Loc3R-VLM**, a novel framework that equips 2D VLMs with advanced 3D spatial understanding capabilities from video. Inspired by human cognition, it builds an internal cognitive map of the global environment while explicitly modeling an agent's position and orientation. By jointly capturing global layout and egocentric state, the model excels at two core tasks: **language-driven localization** and **viewpoint-aware 3D reasoning**.

<p align="center">
  <img src="https://microsoft.github.io/Loc3R-VLM/static/images/teaser.png" width="100%" alt="Loc3R-VLM Overview">
</p>

## 🚧 Code Release

⌛ **We are currently preparing the codebase for release. Stay tuned!**

---

## 💡 Key Features

* **Layout Reconstruction:** Reconstructs the 3D spatial layout from sequential video frames.
* **Localization:** Determines the agent's coordinates and orientation based on natural language descriptions (e.g., *"I am facing the window with a blue cube to my right"*).
* **3D Reasoning:** Answers complex navigation or spatial questions by combining localized state with the global map.

