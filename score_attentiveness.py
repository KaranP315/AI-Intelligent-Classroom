"""
score_attentiveness.py - Final Engagement Score Calculator (Bridge)

This script reads the raw CV and Audio logs, reads the day's subject text,
and passes everything to the robust Double-Gated Time-Window Aggregation
module to generate the final fused report.
"""

import os
import json
import pandas as pd
from datetime import datetime
import double_gated_scorer
import feedback_generator

def read_subject_text(filepath="subject.txt"):
    """Reads the day's class topic from the text file."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Semantic relevance gating will be disabled (all speech will pass).")
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def parse_audio_events(dialogue_log_path, session_start_time=None):
    """
    Reads sscon_output.json and formats it into the expected event structure.
    Returns a dict of {student_name: [event1, event2, ...]}
    """
    try:
        with open(dialogue_log_path, 'r', encoding='utf-8') as f:
            raw_log = json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

    if not raw_log:
        return {}

    # Find the earliest timestamp to act as t=0 if session_start_time isn't provided
    timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in raw_log]
    if not session_start_time and timestamps:
        session_start_time = min(timestamps)

    student_audio = {}
    for entry in raw_log:
        speaker = entry.get('speaker', 'Unknown')
        dt = datetime.fromisoformat(entry['timestamp'])
        
        # Calculate seconds since session start
        t_seconds = (dt - session_start_time).total_seconds() if session_start_time else 0.0
        
        # Determine if SSCon considered it reasoning
        # SSCon acts: "Providing evidence", "Making a claim", "Relating to another student"
        act = entry.get('dialogue_act', '')
        is_reasoning = act in ["Providing evidence", "Making a claim", "Relating to another student"]
        
        event = {
            'timestamp': t_seconds,
            'is_reasoning': is_reasoning,
            'spoken_text': entry.get('text', '')
        }
        
        if speaker not in student_audio:
            student_audio[speaker] = []
        student_audio[speaker].append(event)
        
    return student_audio

def parse_cv_logs(cv_csv_path):
    """
    Reads attentiveness_log.csv and formats it into lists of per-frame behaviors.
    Returns a dict of {student_name: [behavior1, behavior2, ...]} and the session start time.
    """
    try:
        df = pd.read_csv(cv_csv_path)
    except FileNotFoundError:
        return {}, None

    if df.empty:
        return {}, None

    # Sort by timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    session_start_time = df['Timestamp'].iloc[0]
    
    student_cv = {}
    for person in df['Person'].unique():
        # Get behavior list in chronological order
        behaviors = df[df['Person'] == person]['Behavior'].tolist()
        student_cv[person] = behaviors
        
    return student_cv, session_start_time

def generate_final_report(dialogue_log_path, cv_csv_path):
    """
    Coordinates data parsing and calls the Double-Gated Scorer.
    """
    print("\n" + "=" * 60)
    print("        FINAL STUDENT ATTENTIVENESS REPORT")
    print("=" * 60)

    # 1. Load Subject Text
    base_dir = os.path.dirname(os.path.abspath(__file__))
    subject_text = read_subject_text(os.path.join(base_dir, "subject.txt"))
    
    # 2. Parse Raw Logs
    student_cv, session_start_time = parse_cv_logs(cv_csv_path)
    student_audio = parse_audio_events(dialogue_log_path, session_start_time)

    # Gather all unique students
    all_students = set(list(student_cv.keys()) + list(student_audio.keys()))
    
    # If registered students were detected, ignore spurious unassigned 'Unknown' frames
    named_students = {s for s in all_students if s != "Unknown"}
    if named_students:
        all_students = named_students

    if not all_students:
        print("\nNo student data found. Run a session first!")
        return

    # 3. Process each student
    for student in sorted(all_students):
        cv_logs = student_cv.get(student, [])
        audio_events = student_audio.get(student, [])
        
        # 4. Call the robust Double-Gated Scorer!
        result = double_gated_scorer.calculate_final_grade(cv_logs, audio_events, subject_text)

        print(f"\n{'-' * 50}")
        print(f"  Student: {student}")
        print(f"{'-' * 50}")
        print(f"   Physical Attentiveness (CV):      {result['cv_earned']}/{result['cv_max']}  ({result['cv_percentage']}%)")
        print(f"   Dialogue Participation Points:     {result['audio_earned']}/{result['audio_max']}  ({result['audio_percentage']}%)")
        print(f"  ")
        print(f"   OVERALL ENGAGEMENT SCORE:          {result['final_score']}/100")
        
        # Print breakdown of audio gates for transparency
        if audio_events:
            print(f"  \n  [Audio Breakdown]")
            on_topic_count = 0
            off_topic_count = 0
            reasoning_count = 0
            questioning_count = 0
            yes_no_count = 0

            for event in audio_events:
                spoken = event.get('spoken_text', '').strip()
                is_rel = double_gated_scorer.check_audio_relevance(spoken, subject_text)
                is_reason = event.get('is_reasoning', False)

                if is_rel:
                    on_topic_count += 1
                    if is_reason:
                        reasoning_count += 1
                        tier = f"On-Topic Reasoning (+{double_gated_scorer.ON_TOPIC_POINTS} pts)"
                    else:
                        questioning_count += 1
                        tier = f"On-Topic (+{double_gated_scorer.ON_TOPIC_POINTS} pts)"
                else:
                    off_topic_count += 1
                    tier = f"Off-Topic (-{double_gated_scorer.OFF_TOPIC_PENALTY} pts)"

                print(f"   t={int(event['timestamp'])}s | {tier:24} | \"{spoken[:30]}...\"")

            total_spk = on_topic_count + off_topic_count
            plus_pts = on_topic_count * double_gated_scorer.ON_TOPIC_POINTS
            minus_pts = off_topic_count * double_gated_scorer.OFF_TOPIC_PENALTY
            net_pts = max(0, plus_pts - minus_pts)
            max_pts = total_spk * double_gated_scorer.ON_TOPIC_POINTS if total_spk > 0 else 0
            print(f"   Summary: {total_spk} utterances | {on_topic_count} On-Topic (+{plus_pts} pts) | {off_topic_count} Off-Topic (-{minus_pts} pts) | Net: {net_pts}/{max_pts} pts")

            audio_profile = {
                'reasoning_count': reasoning_count,
                'questioning_count': questioning_count,
                'yes_no_count': yes_no_count,
                'off_topic_count': off_topic_count,
            }

            # Generate and print behavioral feedback
            fb = feedback_generator.generate_student_feedback(audio_profile)
            print(f"\n   Teacher Feedback: {fb}")
        else:
            fb = feedback_generator.generate_student_feedback({})
            print(f"\n   Teacher Feedback: {fb}")

    print(f"\n{'=' * 60}\n")

if __name__ == "__main__":
    import sys

    # Default paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cv_path = os.path.join(base_dir, "Intelligent-Classroom-System", "attentiveness_log.csv")
    sscon_path = os.path.join(base_dir, "sscon_output.json")

    if len(sys.argv) > 1:
        cv_path = sys.argv[1]
    if len(sys.argv) > 2:
        sscon_path = sys.argv[2]

    # Preload the sentence transformer model so it doesn't interrupt the output print block
    print("Loading AI Relevance Gateway model...")
    double_gated_scorer._get_model()

    generate_final_report(sscon_path, cv_path)
