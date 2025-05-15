Welcome to the TSC Project Documentation!
==========================================

📌 **Course:** Tools for Scientific Computing (TSC) and Machine Learning for Fluid Dynamics (MLFD)

🏫 **von Karman Institute for Fluid Dynamics**  

🌀  **Environmental and Applied Fluid Dynamics Department**

---

📌 **Project overview:**

This repository contains a series of scripts designed to post-process Particle Image Velocimetry (PIV) data. This project focuses on analyzing and comparing the velocity fields obtained from experimental tests, providing robust visualization tools and data analysis pipelines. The goal is to enable efficient interpretation of flow structures and evaluate experimental results with Proper Orthogonal Decomposition (POD).
The scripts are modular and tailored for large datasets, supporting batch processing of PIV outputs. The main functionalities include:

- Data Import and Preprocessing: Automated loading of PIV velocity fields from TXT files, data cleaning, and preprocessing for consistency.

- PCA-based Comparison: Quantitative comparison of different velocity fields using PCA to highlight the main flow features and differences.

- Flow Visualization: Generation of vector plots, contour maps, and POD modes for qualitative and quantitative flow analysis.

- Automation and Batch Processing: Designed to handle hundreds of PIV files efficiently, supporting iterative analysis workflows.

---

📚 **Code Documentation contents:**

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   autoapi/Project/index

---

🚀 **Author:** Giacomo Gaetano De Sio 

📅 **Last updated:** |today|
