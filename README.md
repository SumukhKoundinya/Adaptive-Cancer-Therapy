 # Adaptive Cancer Therapy Simulation

Adaptive Cancer Therapy is a research-oriented Python project that combines machine learning, clinical simulation, and game-theoretic tumor modeling to explore personalized treatment strategies for glioblastoma.

## What it does
- Loads clinical GBM data and builds ML-ready patient datasets
- Trains and evaluates response and survival prediction models
- Simulates adaptive treatment policies and tumor evolution
- Implements a Δ-Nim inspired tumor game model for therapy planning
- Includes OpenVINO inference support for optimized model deployment

## Key components
- `data/`: dataset loading, preprocessing, and encoding
- `ml/`: feature modeling, response/survival prediction, OpenVINO inference
- `engine/`, `game/`: tumor game logic, transition rules, payoff modeling
- `policy/`: treatment policy algorithms and strategy selection
- `evaluation/`: metrics, model comparison, visualization, ablations
- `simulation/`: patient trajectory simulation and environment setup
- `visualization/`: policy comparison, survival curves, tumor dynamics

## Why it matters
This project demonstrates a novel hybrid approach to cancer therapy research by combining patient data modeling with strategic decision-making. It is useful for exploring how adaptive therapy and tumor response prediction can improve long-term outcomes.

## Setup
1. Clone the repository
2. Create a Python virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements_openvino.txt
   ```
4. Run evaluations and reports:
   ```bash
   python scripts/evaluate_all.py
   ```

## Usage
- `python main.py` for the top-level workflow
- `python scripts/train_models.py` to train models
- `python scripts/evaluate_all.py` to generate evaluation reports
- `demo/simulation.py` or `demo/openvino_demo.py` for demos

## Results
Generated outputs include:
- `results/evaluation_report.csv`
- `results/response_confusion_matrix.csv`
- visualization charts for treatment policies and survival curves

## Skills demonstrated
- Python data engineering and model training
- Predictive modeling with PyTorch and OpenVINO
- Game theory and algorithmic decision support
- Research-oriented experiment evaluation and reporting
- Clinical simulation and visualization

## Notes
This repository is built for academic exploration. The code is structured to make it easy to test, extend, and visualize adaptive therapy strategies for glioblastoma research.
