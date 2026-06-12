from typing import List, Dict, Any
import numpy as np


def aggregate_session(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    emotions = [r["dominant"] for r in records if r.get("dominant")]
    emotion_counts: Dict[str, int] = {}
    for e in emotions:
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    total = len(emotions) or 1
    emotion_dist = {k: round(v / total * 100, 1) for k, v in emotion_counts.items()}

    conf_scores  = [r["confidence_score"] for r in records if "confidence_score" in r]
    stress_scores= [r["stress_score"]     for r in records if "stress_score"      in r]
    att_scores   = [r["attention_score"]  for r in records if "attention_score"   in r]
    eye_scores   = [r["eye_contact_score"]for r in records if "eye_contact_score" in r]

    avg_conf    = round(float(np.mean(conf_scores)),   1) if conf_scores   else 0
    avg_stress  = round(float(np.mean(stress_scores)), 1) if stress_scores else 0
    avg_att     = round(float(np.mean(att_scores)),    1) if att_scores    else 0
    avg_eye     = round(float(np.mean(eye_scores)),    1) if eye_scores    else 0

    stress_level = "High" if avg_stress > 60 else "Medium" if avg_stress > 35 else "Low"
    att_state    = "Focused" if avg_att > 65 else "Distracted"

    recs = []
    if avg_eye < 60:
        recs.append("Maintain more eye contact with the camera to project confidence.")
    if avg_stress > 50:
        recs.append("Practise deep breathing techniques before high-pressure sessions.")
    if avg_att < 60:
        recs.append("Minimise distractions — face the camera directly and stay engaged.")
    if emotion_dist.get("Angry", 0) > 20:
        recs.append("Work on keeping a calm and positive expression during interviews.")
    if emotion_dist.get("Happy", 0) < 20:
        recs.append("Smile naturally to create a positive impression on interviewers.")
    if not recs:
        recs.append("Excellent performance! Keep maintaining your composure and confidence.")

    return dict(
        emotion_distribution=emotion_dist,
        avg_confidence=avg_conf,
        avg_stress=avg_stress,
        avg_attention=avg_att,
        avg_eye_contact=avg_eye,
        stress_level=stress_level,
        attention_state=att_state,
        dominant_emotion=max(emotion_counts, key=emotion_counts.get) if emotion_counts else "Neutral",
        recommendations=recs,
        total_seconds=len(records),
    )
