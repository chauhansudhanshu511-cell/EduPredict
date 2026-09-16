"""Support bands and rule-based study suggestions.

Both are simple, transparent RULES written by hand. They are not produced by
the machine-learning model and are not validated predictions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import DEFAULT_STRONG_THRESHOLD, DEFAULT_SUPPORT_THRESHOLD


@dataclass(frozen=True)
class SupportBand:
    key: str
    label: str
    color: str
    message: str


BANDS = {
    "support": SupportBand("support", "Additional support suggested", "#C2410C",
                           "A check-in with the instructor or tutor could help plan the next weeks."),
    "progressing": SupportBand("progressing", "Progressing", "#B45309",
                               "On track overall, with room to strengthen a few habits."),
    "strong": SupportBand("strong", "Strong progress", "#047857",
                          "Current habits look effective. Keep them consistent until the exam."),
}


def support_band(score: float, support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
                 strong_threshold: float = DEFAULT_STRONG_THRESHOLD) -> SupportBand:
    if support_threshold >= strong_threshold:
        raise ValueError("support_threshold must be lower than strong_threshold")
    if score < support_threshold:
        return BANDS["support"]
    if score < strong_threshold:
        return BANDS["progressing"]
    return BANDS["strong"]


def _present(values: dict, key: str) -> float | None:
    v = values.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def study_suggestions(values: dict, max_items: int = 5) -> list[str]:
    """Return practical suggestions triggered by simple thresholds on the inputs."""
    tips: list[tuple[int, str]] = []  # (priority, text); lower = more important

    missed = _present(values, "missed_assignments")
    if missed is not None and missed >= 3:
        tips.append((1, f"{missed:.0f} assignments are missing. Ask whether any can still be submitted "
                        "and set a weekly deadline reminder."))
    elif missed is not None and missed >= 1:
        tips.append((4, "Catch up on the missed assignment(s); they usually cover exam topics."))

    attendance = _present(values, "attendance_pct")
    if attendance is not None and attendance < 75:
        tips.append((1, f"Attendance is {attendance:.0f}%. Aim to attend every remaining class and "
                        "get notes for sessions already missed."))
    elif attendance is not None and attendance < 90:
        tips.append((5, "Attending all remaining classes helps with revision topics and exam hints."))

    hours = _present(values, "study_hours_per_week")
    if hours is not None and hours < 5:
        tips.append((2, f"About {hours:.0f} study hours per week is low. Try adding short, scheduled "
                        "sessions (e.g. 30-45 minutes on most days)."))

    prev = _present(values, "previous_exam_score")
    if prev is not None and prev < 60:
        tips.append((2, "Review the previous exam: list the questions lost and re-practise those topics."))

    quiz = _present(values, "quiz_avg")
    if quiz is not None and quiz < 65:
        tips.append((3, "Quiz results suggest gaps in recall. Use practice questions and self-testing, "
                        "not only re-reading."))

    assignment = _present(values, "assignment_avg")
    if assignment is not None and assignment < 65:
        tips.append((3, "Use assignment feedback: ask the instructor to explain the biggest mark losses."))

    participation = _present(values, "participation_score")
    if participation is not None and participation < 4:
        tips.append((4, "Ask at least one question per week in class or office hours to clarify doubts early."))

    missing = [k for k in ("attendance_pct", "study_hours_per_week", "previous_exam_score",
                           "assignment_avg", "quiz_avg", "missed_assignments", "participation_score")
               if _present(values, k) is None]
    if missing:
        tips.append((6, "Some inputs were left blank, so the estimate relies more on typical values."))

    if not tips:
        tips.append((9, "Keep the current routine and add a timed practice exam in the final two weeks."))

    tips.sort(key=lambda t: t[0])
    return [text for _, text in tips[:max_items]]
