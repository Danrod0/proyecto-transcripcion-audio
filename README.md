# Uncertainty Estimation for Audio Transcription Models

This repository contains the code, Jupyter notebooks, and experimental results from a research project focused on **uncertainty estimation in automatic speech transcription**, using **OpenAI Whisper** as the base model.

The main objective of the project is to evaluate and compare the performance of three uncertainty estimation methods:

1. **Monte Carlo Dropout**
2. **Temperature Scaling**
3. **Feature Densities**

Each method is implemented through structured Jupyter notebooks that include:

- Model loading (Whisper and variants)  
- Data loading and preprocessing  
- Training or calibration steps  
- Evaluation on speech datasets  
- Metric computation (WER, correlations, calibration metrics)  
- Result visualization and comparison  

The repository is organized to support reproducible research and collaborative development among researchers and student assistants.

---

## 📁 Repository Structure

proyecto-transcripcion-audio-final/
├── data/ # Raw, processed, and example datasets
├── notebooks/ # Jupyter notebooks grouped by uncertainty method
├── src/ # Modular Python source code
├── results/ # Metrics, plots, logs, and outputs
└── docs/ # Methodology, documentation, and references

Each directory includes its own `README.md` to guide users.

---

## 🚀 Installation

### Option 1: Python + pip
pip install -r requirements.txt

### Option 2: Conda (recommended)
conda env create -f environment.yml
conda activate whisper-uncertainty

---

## ▶️ Running the Notebooks

The notebooks are organized by uncertainty estimation technique:

notebooks/
├── uncertainty-mc-dropout/
├── temperature-scaling/
└── feature-densities/

Each subfolder includes:

- The main notebook(s) for experiments  
- Auxiliary notebooks when needed  
- A `README.md` with method-specific instructions  

---

Results include:

- Correlation between uncertainty and transcription error  
- Word Error Rate (WER)   
- Visual comparisons among uncertainty estimation methods  

---

## 📚 Documentation

The `docs/` directory includes extended documentation:

- Methodology and experiment design  
- Pipeline architecture  
- Notes for collaborators  
- Bibliography and references  

---

## 👥 Research Team

- Principal investigators: *Dr. Saúl Calderón-Ramírez*, *Dr. Martín Solís*, *M.Sc Walter Morales-Muñoz*, *M.Sc J. Esteban Pérez-Hidalgo*  
- Student assistants: *Daniel Rodríguez*, *Erick Muñoz*
- Institution: *Technological Institute of Costa Rica*  

---

## 📜 License

This project is distributed under the MIT License.

---

If you have questions about reproducing the experiments or running the models, feel free to open an issue or contact the research team.
