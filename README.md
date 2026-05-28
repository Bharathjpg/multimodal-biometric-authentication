# Multimodal Biometric Authentication System
**Research Project III — Bharath Neelakrishnan — VMU, 2026**

## Overview
A locking and unlocking decision system based on person identity recognition,
combining face recognition, voice authentication, and knowledge-based
authentication (KBA) with weighted score-level fusion.

## System Pipeline
Face Module → Voice Module → KBA Module → WSR Fusion → Access Decision

## Algorithms Compared
- LBPH (radius=1, neighbors=8)
- Eigenfaces (40 components)
- Fisherfaces (39 components)

## Datasets Used
- AT&T (ORL): 40 subjects, 400 images
- Extended Yale-B: 38 subjects, 2,414 images
- LFW Restricted: 236 subjects, 19,038 images
- **Total: 314 unique identities, 21,852 images**

## Key Results
| Algorithm   | Accuracy | FAR  | FRR  |
|-------------|----------|------|------|
| LBPH        | 99.5%    | 1.2% | 0.5% |
| Eigenfaces  | 96.8%    | 2.1% | 3.1% |
| Fisherfaces | 94.0%    | 3.2% | 2.8% |
| WSR Fusion  | 99.7%    | 0.8% | 0.3% |

## How to Run
```bash
pip install -r requirements.txt
python src/main.py
```

## Project Structure
src/
├── face/        - Face recognition module (LBPH, Eigenfaces, Fisherfaces)
├── voice/       - Voice authentication module (MFCC)
├── kba/         - Knowledge-based authentication module
├── fusion/      - Weighted Score Rule fusion
└── main.py      - Full pipeline entry point
datasets/        - Dataset download instructions
results/         - Experimental results

## Technologies Used
- Python 3.11
- OpenCV 4.8 (face recognition)
- librosa (MFCC voice features)
- sounddevice (audio capture)
- scipy, numpy, scikit-learn

## Author
Bharath Neelakrishnan | MIF240018 | VMU 2026
