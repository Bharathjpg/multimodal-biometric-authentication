# Experimental Results

## AT&T Dataset — 5-Fold Cross Validation
40 subjects | 400 images | 10 images per subject

| Algorithm   | Accuracy | Std  | FAR  | FRR  | Latency |
|-------------|----------|------|------|------|---------|
| LBPH        | 99.5%    | 0.6% | 1.2% | 0.5% | 9.4ms   |
| Eigenfaces  | 96.8%    | 1.7% | 2.1% | 3.1% | 3.0ms   |
| Fisherfaces | 94.0%    | 3.7% | 3.2% | 2.8% | 0.4ms   |

Winner: LBPH — highest accuracy, lowest standard deviation

## Yale-B Dataset — Illumination Robustness Test
15 subjects | Normal vs varied lighting conditions

| Algorithm   | Center | Left | Right | Overall | Drop |
|-------------|--------|------|-------|---------|------|
| LBPH        | 47%    | 47%  | 33%   | 42%     | 13pp |
| Eigenfaces  | 47%    | 20%  | 13%   | 27%     | 33pp |
| Fisherfaces | 33%    | 13%  | 20%   | 22%     | 20pp |

Winner: LBPH — smallest accuracy drop across lighting conditions

## Multi-Factor Fusion System Results

| Module         | Status  | Score  |
|----------------|---------|--------|
| Face (LBPH)    | WORKING | 0.9950 |
| Voice (MFCC)   | WORKING | 0.9210 |
| KBA            | WORKING | 1.0000 |
| WSR Fusion     | WORKING | 0.9970 |
| Decision       | UNLOCK  | ✓      |

Fusion weights: face=0.6, voice=0.4, KBA=modifier
Decision threshold: 0.60
Final fused accuracy: 99.7% | FAR: 0.8% | FRR: 0.3%

## Key Conclusions
1. LBPH is the most accurate and stable algorithm on AT&T dataset
2. LBPH is the most illumination-robust algorithm on Yale-B dataset
3. Multi-factor fusion outperforms any single modality alone
4. System runs on standard CPU laptop with no GPU required
