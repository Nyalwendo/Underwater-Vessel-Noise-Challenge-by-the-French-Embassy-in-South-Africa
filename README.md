# Underwater-Vessel-Noise-Challenge-by-the-French-Embassy-in-South-Africa
Develop machine learning models to detect underwater vessel noise from unlabelled South African hydrophone data. Predict vessel energy fractions and evidence scores in controlled tests, and generate relative acoustic activity pressure maps.
SA Coastal Vessel Acoustic Monitor & URN Detection Engine
1. What This Project Does
This project provides an automated machine learning and acoustic processing pipeline designed to monitor Underwater Radiated Noise (URN) along the South African coastline.

Operating on unlabelled one-minute frequency-band summaries supplied by SAEON, SANBI, and the South African Underwater Radiated Noise Consortium (SA URN), the system parses log-spaced third-octave sound energy levels (ranging from ~31.6 Hz to 7.94 kHz). It distinguishes vessel propulsion signatures from natural ambient sounds (wind, waves, and biological activity), predicts vessel energy contribution fractions, evaluates acoustic detection confidence scores, and builds comprehensive spatial pressure maps tracking noise variation across marine monitoring sites over time.

2. Why Is It Useful?
Marine Ecosystem Protection: Commercial shipping and small boat engines produce underwater noise that disrupts marine animals relying on sound for navigation, feeding, and communication. This toolkit helps quantify and mitigate those impacts.

Unlabelled Data Adaptation: Because real-world coastal hydrophone recordings lack verified manual vessel tags, the framework uses robust feature engineering and controlled-addition testing methodologies to provide reliable target comparisons.

Spatial Pressure Mapping: Researchers can visualize relative acoustic activity across different coastal recording sites to inform marine spatial planning and policy decisions.

3. Installation and Setup (Step-by-Step)
Follow these steps to set up your local environment and run the pipeline:

Step 1: Clone the Repository
Bash
git clone https://github.com/Nyalwendo/sa-coastal-vessel-noise.git
cd sa-coastal-vessel-noise
Step 2: Create and Activate a Python Virtual Environment
Bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Step 3: Install Dependencies
Bash
pip install --upgrade pip
pip install numpy pandas scikit-learn torch torchvision pyarrow matplotlib seaborn
Step 4: Configure Data Layout
Download the dataset files from the data challenge source and structure your workspace directory as follows:

Plaintext
project_root/
│
├── data/
│   ├── acoustics/         # Contains site-specific Parquet files
│   ├── stations.csv       # Station coordinates and metadata
│   ├── recorders.csv      # Recorder configuration coverage
│   └── Test.csv           # Evaluation window records
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── mapper.py
│
└── README.md
4. Usage Examples & Code Snippets
Python Code Example: Loading and Preprocessing Acoustic Parquet Data
Python
import pandas as pd
import numpy as np

def load_site_acoustics(parquet_path):
    """Loads and inspects one-minute sound energy summaries from site parquet files."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded records: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Fill or inspect missing observations safely
    band_columns = [col for col in df.columns if 'tol_' in col.lower() or 'band' in col.lower()]
    return df, band_columns

# Example usage:
# df, bands = load_site_acoustics('data/acoustics/site_alpha.parquet')
# print(df.head())
Python Code Example: Model Inference & Energy Fraction Prediction
Python
import torch
import torch.nn as nn

class URNPredictor(nn.Module):
    def __init__(self, input_dim=375):
        super(URNPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(165, 2) # Outputs: [Energy Fraction (0-1), Detection Confidence Score (0-1)]
        )
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))

# Quick verification of model architecture
model = URNPredictor()
print("Initialized URN Neural Network Pipeline successfully.")
Hydrophone Deployment & System Architecture
Plaintext
  [Underwater Hydrophones] 
           │
           ▼ (1-Minute Acoustic Summaries)
  [Band-Level Feature Extraction] 
           │
           ├──────────────────────────────┐
           ▼                              ▼
  [Vessel Energy Fraction Model]   [Evidence Detection Scoring]
           │                              │
           └──────────────┬───────────────┘
                          ▼
             [Spatial Pressure Mapping]
