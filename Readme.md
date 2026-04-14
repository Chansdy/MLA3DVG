# MLA3DVG: Dual-Modality Multi-level Semantic Alignment for Robust Monocular 3D Visual Grounding


---

We introduce **Loc3R-VLM**, a novel framework that equips 2D VLMs with advanced 3D spatial understanding capabilities from video. Inspired by human cognition, it builds an internal cognitive map of the global environment while explicitly modeling an agent's position and orientation. By jointly capturing global layout and egocentric state, the model excels at two core tasks: **language-driven localization** and **viewpoint-aware 3D reasoning**.

<p align="center">
  <img src="https://microsoft.github.io/Loc3R-VLM/static/images/teaser.png" width="100%" alt="MLA3DVG Overview">
</p>

## 🚧 Code Release

⌛ **We are currently preparing the codebase for release. Stay tuned!**

---

## 💡 Key Features

* **Layout Reconstruction:** Reconstructs the 3D spatial layout from sequential video frames.
* **Localization:** Determines the agent's coordinates and orientation based on natural language descriptions (e.g., *"I am facing the window with a blue cube to my right"*).
* **3D Reasoning:** Answers complex navigation or spatial questions by combining localized state with the global map.

