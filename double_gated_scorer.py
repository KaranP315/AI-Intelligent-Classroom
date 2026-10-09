"""
double_gated_scorer.py — Double-Gated Time-Window Aggregation System

Calculates a student's final engagement score by fusing two independent
data streams (Computer Vision + Audio NLP) through 1-minute bucket
aggregation with a semantic relevance gateway.

Architecture:
    ┌------------------------------------------------------------─┐
    │                    session_runner.py                        │
    │          (feeds CV frames + Audio events in real-time)      │
    └--------------┬--------------------------┬------------------─┘
                   │                          │
         CV Frame Logs                  Audio Events
                   │                          │
                   ▼                          ▼
       ┌------------------─┐      ┌------------------------─┐
       │  1-Min CV Buckets │      │  Relevance Gateway      │
       │  (Majority Vote)  │      │  (Sentence Transformers │
       │                   │      │   + Cosine Similarity)  │
       │  write  → 5 pts   │      │                         │
       │  listen → 4 pts   │      │  Gate 1: is_reasoning?  │
       │  distracted → 1pt │      │  Gate 2: on-topic?      │
       └--------┬----------┘      └----------─┬------------─┘
                │                             │
                ▼                             ▼
          CV % (0-100)                  Audio % (0-100)
                │                             │
                └----------┬------------------┘
                           │
                           ▼
                 Final = (CV × 0.60) + (Audio × 0.40)
"""

import math
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------─
# Lazy-loaded Singleton for the Sentence Transformer model.
# This ensures the ~80MB model is loaded into GPU/CPU memory exactly ONCE,
# no matter how many times check_audio_relevance() is called.
# ----------------------------------------------------------------------------─

_model = None


def _get_model():
    """
    Returns the cached SentenceTransformer instance.
    On the first call, it downloads/loads 'all-MiniLM-L6-v2' (~80MB).
    All subsequent calls return the same object (zero overhead).
    """
    global _model
    if _model is None:
        import os
        from sentence_transformers import SentenceTransformer
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(base_dir, "all-MiniLM-L6-v2")
        if os.path.exists(local_path):
            _model = SentenceTransformer(local_path)
        else:
            _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Audio Relevance Gateway
# ═════════════════════════════════════════════════════════════════════════════

def check_audio_relevance(spoken_text: str, subject_text: str, threshold: float = 0.25) -> bool:
    """
    Semantic relevance gate. Compares a student's spoken utterance against
    the day's uploaded subject/lecture text using cosine similarity of
    sentence embeddings.
    
    Evaluates similarity against both the full text and individual concept
    sentences, taking the maximum score to accurately match short or conversational
    classroom questions.

    Args:
        spoken_text:  What the student actually said (transcribed by Whisper).
        subject_text: The reference text for today's class topic/lecture notes.
        threshold:    Minimum cosine similarity to be considered "on-topic" (default: 0.25).

    Returns:
        True if the utterance is semantically relevant to the subject matter.
    """
    if not spoken_text or not spoken_text.strip():
        return False
    if not subject_text or not subject_text.strip():
        return True

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = _get_model()

    # Split subject text into sentences/clauses for granular semantic matching
    sentences = [
        s.strip() for s in subject_text.replace('\n', '. ').split('.')
        if len(s.strip()) > 5
    ]
    all_refs = [subject_text.strip()] + (sentences if sentences else [])

    spoken_emb = model.encode([spoken_text.strip()])
    refs_emb = model.encode(all_refs)

    sims = cosine_similarity(spoken_emb, refs_emb)[0]
    max_similarity = float(np.max(sims))

    return max_similarity >= threshold


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2a: CV 1-Minute Bucket Aggregator
# ═════════════════════════════════════════════════════════════════════════════

# Points awarded for the DOMINANT behavior in each 1-minute window.
# Max possible per bucket = 5 (achieved by 'write').
# These keys MUST match the labels from the CV model (config.py class_names).
CV_BEHAVIOR_POINTS = {
    'write':  5,    # Actively taking notes
    'listen': 4,    # Paying attention to the board/teacher
    'drink':  1,    # Mild distraction (drinking water, etc.)
    'trance': 0,    # Completely zoned out
}


def calculate_cv_points(cv_frame_logs: list) -> tuple:
    """
    Aggregates raw frame-by-frame CV behavior labels into 1-minute buckets
    using majority voting (statistical mode), then assigns points.

    Args:
        cv_frame_logs: A flat list of behavior strings, one per frame/second.
                       e.g. ['listening', 'listening', 'writing', ...]
                       Every 60 consecutive items form one 1-minute bucket.

    Returns:
        (earned_points, max_possible_points)

    Example:
        >>> calculate_cv_points(['listening'] * 50 + ['distracted'] * 10)
        (4, 5)   # 1 bucket, mode = 'listening' → 4 pts, max = 5
    """
    if not cv_frame_logs:
        return (0, 0)

    total_frames = len(cv_frame_logs)
    num_buckets = math.ceil(total_frames / 60)  # partial last bucket still counts

    earned = 0
    for bucket_idx in range(num_buckets):
        start = bucket_idx * 60
        end = min(start + 60, total_frames)
        chunk = cv_frame_logs[start:end]

        if not chunk:
            continue

        # Majority vote: find the most common behavior in this 60s window
        counter = Counter(chunk)
        dominant_behavior = counter.most_common(1)[0][0]

        # Map to points (unknown behaviors default to 0)
        earned += CV_BEHAVIOR_POINTS.get(dominant_behavior, 0)

    max_possible = num_buckets * 5
    return (earned, max_possible)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2b: Audio Participation Scoring (Utterance-Level)
# ═════════════════════════════════════════════════════════════════════════════

ON_TOPIC_POINTS = 4
OFF_TOPIC_PENALTY = 2  # Deducts 2 points per off-topic utterance


def calculate_audio_points(audio_events: list, subject_text: str,
                           session_buckets: int = 0,
                           off_topic_penalty: int = None) -> tuple:
    """
    Calculates audio participation points based on utterance-level relevance:
      - Each on-topic utterance: +4 points
      - Each off-topic utterance: -penalty points (default: -2)
      - If net points <= 0: earned = 0.0, audio_pct = 0.0%
      - If net points > 0: audio_pct = (net_points / (total_spoken * 4)) * 100
      - Scaled out of 10: earned = round((audio_pct / 100.0) * 10.0, 1)

    Args:
        audio_events: List of dicts, each containing:
            {
                'timestamp': float,      # seconds since session start
                'is_reasoning': bool,    # SSCon classifier output
                'spoken_text': str       # Whisper transcription
            }
        subject_text: The reference text for today's class topic.
        session_buckets: (Unused / backwards compatible)
        off_topic_penalty: Deduction per off-topic utterance (default: 2)

    Returns:
        (earned_out_of_10, max_10)
    """
    if not audio_events:
        return (0.0, 10)

    if off_topic_penalty is None:
        import os
        off_topic_penalty = int(os.getenv("OFF_TOPIC_PENALTY", str(OFF_TOPIC_PENALTY)))

    on_topic_count = 0
    off_topic_count = 0

    for event in audio_events:
        spoken = event.get('spoken_text', '').strip()
        if not spoken:
            continue
        is_relevant = check_audio_relevance(spoken, subject_text)
        if is_relevant:
            on_topic_count += 1
        else:
            off_topic_count += 1

    total_spoken = on_topic_count + off_topic_count
    if total_spoken == 0:
        return (0.0, 10)

    plus_points = on_topic_count * ON_TOPIC_POINTS
    minus_points = off_topic_count * off_topic_penalty
    net_points = plus_points - minus_points

    # If in total it is negative, put 0 on 10
    if net_points <= 0:
        earned_out_of_10 = 0.0
    else:
        max_possible_points = total_spoken * ON_TOPIC_POINTS
        # (plus points divided by each time spoken * 4) * 100
        audio_pct = (net_points / max_possible_points) * 100.0
        earned_out_of_10 = round((audio_pct / 100.0) * 10.0, 1)

    return (earned_out_of_10, 10)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Final Score Blender
# ═════════════════════════════════════════════════════════════════════════════

def calculate_final_grade(cv_frame_logs: list, audio_events: list,
                          subject_text: str) -> dict:
    """
    Fuses CV and Audio scores into a single weighted engagement grade.

    Blend formula:  Final = (CV% × 0.60) + (Audio% × 0.40)

    Args:
        cv_frame_logs: List of per-second behavior labels.
        audio_events:  List of audio event dicts (see calculate_audio_points).
        subject_text:  Today's class subject/lecture text.

    Returns:
        dict with all raw totals, percentages, and the final combined score.

    Example output:
        {
            'cv_earned': 17,
            'cv_max': 25,
            'cv_percentage': 68.0,
            'audio_earned': 15,
            'audio_max': 25,
            'audio_percentage': 60.0,
            'cv_weight': 0.60,
            'audio_weight': 0.40,
            'final_score': 64.8
        }
    """
    # -- CV Stream --
    cv_earned, cv_max = calculate_cv_points(cv_frame_logs)
    cv_pct = (cv_earned / cv_max * 100) if cv_max > 0 else 0.0

    # Determine how many 1-minute buckets the session spans (from CV frames)
    cv_num_buckets = math.ceil(len(cv_frame_logs) / 60) if cv_frame_logs else 0

    # -- Audio Stream --
    # Pass the CV bucket count so audio uses the SAME session length
    audio_earned, audio_max = calculate_audio_points(
        audio_events, subject_text, session_buckets=cv_num_buckets
    )
    audio_pct = (audio_earned / audio_max * 100) if audio_max > 0 else 0.0

    # -- Weighted Blend --
    # If a stream has zero data (max == 0), that stream's weight shifts
    # entirely to the other stream so we don't penalize missing hardware.
    if cv_max == 0 and audio_max == 0:
        final = 0.0
        cv_weight, audio_weight = 0.0, 0.0
    elif cv_max == 0:
        # No camera data — score entirely on audio
        final = audio_pct
        cv_weight, audio_weight = 0.0, 1.0
    elif audio_max == 0:
        # No audio data — score entirely on CV
        final = cv_pct
        cv_weight, audio_weight = 1.0, 0.0
    else:
        # Normal case: both streams available
        cv_weight, audio_weight = 0.60, 0.40
        final = (cv_pct * cv_weight) + (audio_pct * audio_weight)

    return {
        'cv_earned':        cv_earned,
        'cv_max':           cv_max,
        'cv_percentage':    round(cv_pct, 2),
        'audio_earned':     audio_earned,
        'audio_max':        audio_max,
        'audio_percentage': round(audio_pct, 2),
        'cv_weight':        cv_weight,
        'audio_weight':     audio_weight,
        'final_score':      round(final, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Demo / Self-Test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  DOUBLE-GATED TIME-WINDOW AGGREGATION — SELF-TEST")
    print("=" * 65)

    # -- Simulated 5-minute session for student "Sam" --
    subject = (
        "Newton's laws of motion. An object at rest stays at rest unless "
        "acted on by an external force. F equals m times a. Gravity pulls "
        "objects toward Earth at 9.8 m/s squared."
    )

    # CV: 5 minutes = 300 frames (1 per second)
    cv_logs = (
        ['writing'] * 60 +      # Minute 1: writing (5 pts)
        ['listening'] * 60 +     # Minute 2: listening (4 pts)
        ['listening'] * 45 + ['distracted'] * 15 +  # Minute 3: listening wins (4 pts)
        ['distracted'] * 60 +    # Minute 4: distracted (1 pt)
        ['writing'] * 60         # Minute 5: writing (5 pts)
    )

    # Audio: events scattered across 5 minutes
    audio = [
        # Minute 0 (0-59s): On-topic reasoning → should PASS both gates
        {'timestamp': 25, 'is_reasoning': True,
         'spoken_text': 'I think gravity accelerates at 9.8 meters per second squared'},

        # Minute 1 (60-119s): Reasoning but OFF-TOPIC → should FAIL gate 2
        {'timestamp': 90, 'is_reasoning': True,
         'spoken_text': 'I had pizza for lunch and it was amazing'},

        # Minute 2 (120-179s): On-topic but NOT reasoning → should FAIL gate 1
        {'timestamp': 150, 'is_reasoning': False,
         'spoken_text': 'Force equals mass times acceleration'},

        # Minute 3 (180-239s): Silent — no events → 0 pts

        # Minute 4 (240-299s): On-topic reasoning → should PASS both gates
        {'timestamp': 270, 'is_reasoning': True,
         'spoken_text': 'Newton said an object at rest stays at rest'},
    ]

    print(f"\n[Subject] Subject: \"{subject[:60]}...\"")
    print(f"[CV] CV Frames: {len(cv_logs)} (= {len(cv_logs)//60} minutes)")
    print(f"[Audio] Audio Events: {len(audio)}")

    # -- Run the relevance gateway on each audio event for visibility --
    print("\n-- Audio Relevance Gateway Results --")
    for evt in audio:
        relevant = check_audio_relevance(evt['spoken_text'], subject)
        reasoning = evt['is_reasoning']
        gate = "PASS" if (reasoning and relevant) else "FAIL"
        print(f"  [{gate}] t={evt['timestamp']:>3}s | "
              f"reasoning={reasoning} | relevant={relevant} | "
              f"\"{evt['spoken_text'][:50]}\"")

    # -- Calculate final grade --
    result = calculate_final_grade(cv_logs, audio, subject)

    print("\n-- Score Breakdown --")
    print(f"  [CV] CV:    {result['cv_earned']}/{result['cv_max']} "
          f"= {result['cv_percentage']}%  (weight: {result['cv_weight']})")
    print(f"  [Audio] Audio: {result['audio_earned']}/{result['audio_max']} "
          f"= {result['audio_percentage']}%  (weight: {result['audio_weight']})")
    print(f"\n  [FINAL] FINAL ENGAGEMENT SCORE: {result['final_score']}/100")
    print("=" * 65)
