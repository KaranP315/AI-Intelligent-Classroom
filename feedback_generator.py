"""
feedback_generator.py - Automated Student Behavioral Feedback Generator

Takes a dictionary of audio classification counts from a class session
and runs them through logical thresholds to produce a synthesized
feedback string for a teacher's dashboard.
"""


def generate_student_feedback(audio_profile):
    """
    Generates a human-readable feedback string based on a student's
    audio participation profile from a single class session.

    Args:
        audio_profile: dict with integer counts for:
            - 'reasoning_count':   Times the student used analytical reasoning
            - 'questioning_count': Times the student asked questions
            - 'yes_no_count':      Times the student gave brief acknowledgments
            - 'off_topic_count':   Times the student spoke off-topic

    Returns:
        A synthesized feedback string for the teacher's dashboard.
    """
    reasoning   = audio_profile.get('reasoning_count', 0)
    questioning = audio_profile.get('questioning_count', 0)
    yes_no      = audio_profile.get('yes_no_count', 0)
    off_topic   = audio_profile.get('off_topic_count', 0)

    # Calculate total speaking events across all categories
    total_events = reasoning + questioning + yes_no + off_topic

    # ── Rule 1: Zero Participation ──────────────────────────────────────
    # If the student never spoke at all, return immediately.
    if total_events == 0:
        return ("Silent Observer: This student did not speak during class. "
                "Check CV logs for passive focus.")

    # ── Evaluate threshold rules ────────────────────────────────────────
    # Thresholds use BOTH absolute counts AND proportions so they work
    # for short demos (2-3 utterances) and full classes (30+ utterances).
    feedback_notes = []

    # Rule 2: The "Inquisitive" Check
    # Absolute: asked 5+ questions, OR Proportional: >40% of speech was questions
    if questioning > 4 or (total_events >= 2 and questioning / total_events > 0.40):
        feedback_notes.append(
            "Highly Inquisitive: Frequently asks questions, "
            "showing a strong desire to clarify concepts."
        )

    # Rule 3: The "Analytical" Check
    # Absolute: 4+ reasoning statements, OR Proportional: >40% was reasoning
    if reasoning > 3 or (total_events >= 2 and reasoning / total_events > 0.40):
        feedback_notes.append(
            "Strong Analytical Skills: Frequently uses logical reasoning "
            "and explains concepts well."
        )

    # Rule 4: The "Surface-Level" Check
    # Student talks, but mostly just says "yes/no" rather than reasoning or asking
    if yes_no > (questioning + reasoning):
        feedback_notes.append(
            "Passive Participant: Participates in class, but mostly with "
            "brief, surface-level answers. May need prompting to elaborate."
        )

    # Rule 5: The "Distracted" Check
    # Absolute: 3+ off-topic, OR Proportional: >50% of speech was off-topic
    if off_topic > 2 or (total_events >= 2 and off_topic / total_events > 0.50):
        feedback_notes.append(
            "Distracted Speaker: Tends to participate, but often steers "
            "the conversation off-topic."
        )

    # ── Format the output ───────────────────────────────────────────────
    # If the student spoke but hit NONE of the specific thresholds,
    # return a neutral default feedback string.
    if not feedback_notes:
        return ("Standard Participant: Engages with the class at an "
                "average, expected level.")

    # Otherwise, join all triggered feedback notes with a separator
    return " | ".join(feedback_notes)


# ═════════════════════════════════════════════════════════════════════════════
# Self-Test
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 65)
    print("  FEEDBACK GENERATOR — SELF-TEST")
    print("=" * 65)

    # Test Profile 1: Highly Analytical Student
    profile_analytical = {
        'reasoning_count': 6,
        'questioning_count': 5,
        'yes_no_count': 1,
        'off_topic_count': 0,
    }
    print(f"\n📘 Profile: Analytical Student")
    print(f"   Input:    {profile_analytical}")
    print(f"   Feedback: {generate_student_feedback(profile_analytical)}")

    # Test Profile 2: Silent Student
    profile_silent = {
        'reasoning_count': 0,
        'questioning_count': 0,
        'yes_no_count': 0,
        'off_topic_count': 0,
    }
    print(f"\n📘 Profile: Silent Student")
    print(f"   Input:    {profile_silent}")
    print(f"   Feedback: {generate_student_feedback(profile_silent)}")

    # Test Profile 3: Distracted & Surface-Level Student
    profile_distracted = {
        'reasoning_count': 1,
        'questioning_count': 0,
        'yes_no_count': 5,
        'off_topic_count': 4,
    }
    print(f"\n📘 Profile: Distracted Student")
    print(f"   Input:    {profile_distracted}")
    print(f"   Feedback: {generate_student_feedback(profile_distracted)}")

    # Test Profile 4: Average Student (triggers no specific threshold)
    profile_average = {
        'reasoning_count': 2,
        'questioning_count': 2,
        'yes_no_count': 1,
        'off_topic_count': 1,
    }
    print(f"\n📘 Profile: Average Student")
    print(f"   Input:    {profile_average}")
    print(f"   Feedback: {generate_student_feedback(profile_average)}")

    print(f"\n{'=' * 65}\n")
