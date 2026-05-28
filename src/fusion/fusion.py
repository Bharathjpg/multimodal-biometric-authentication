import numpy as np

# ── WEIGHTED SCORE RULE FUSION ─────────────────────────────
# Weights based on empirical accuracy:
# Face (LBPH): 0.6  — highest accuracy (99.5%)
# Voice (MFCC): 0.4 — second modality
# KBA: used as gate — must pass minimum threshold

def wsr_fusion(face_score, voice_score, kba_score,
               w_face=0.6, w_voice=0.4, theta=0.6):
    """
    Weighted Score Rule Fusion.

    Parameters:
        face_score  : float [0, 1] — normalized face recognition score
        voice_score : float [0, 1] — normalized voice authentication score
        kba_score   : float [0, 1] — knowledge-based authentication score
        w_face      : float — weight for face module (default 0.6)
        w_voice     : float — weight for voice module (default 0.4)
        theta       : float — decision threshold (default 0.6)

    Returns:
        decision    : str — 'UNLOCK' or 'LOCK'
        fused_score : float — final combined score
    """

    # KBA acts as a gate — if completely failed, deny immediately
    if kba_score == 0.0:
        print("  KBA failed — access denied immediately.")
        return 'LOCK', 0.0

    # Weighted sum of face and voice scores
    fused_score = (w_face * face_score) + (w_voice * voice_score)

    # Apply KBA as a modifier (boosts or reduces the fused score)
    fused_score = fused_score * (0.7 + 0.3 * kba_score)

    fused_score = round(float(np.clip(fused_score, 0.0, 1.0)), 4)

    decision = 'UNLOCK' if fused_score >= theta else 'LOCK'

    return decision, fused_score


# ── DISPLAY RESULT ─────────────────────────────────────────
def display_result(face_score, voice_score, kba_score,
                   fused_score, decision):
    print("\n" + "=" * 50)
    print("  AUTHENTICATION RESULT")
    print("=" * 50)
    print(f"  Face score  : {face_score:.4f}")
    print(f"  Voice score : {voice_score:.4f}")
    print(f"  KBA score   : {kba_score:.4f}")
    print(f"  Fused score : {fused_score:.4f}")
    print(f"  Decision    : {decision}")
    print("=" * 50)


# ── MIN-MAX NORMALIZATION ──────────────────────────────────
def minmax_normalize(score, min_val=0.0, max_val=1.0):
    return round(float(np.clip(
        (score - min_val) / (max_val - min_val), 0.0, 1.0)), 4)
