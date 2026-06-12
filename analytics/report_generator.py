import os
import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

DARK_BG   = colors.HexColor("#0D1117")
CARD_BG   = colors.HexColor("#161B22")
ACCENT    = colors.HexColor("#58A6FF")
ACCENT2   = colors.HexColor("#7EE787")
WARN      = colors.HexColor("#F78166")
TEXT_MAIN = colors.HexColor("#E6EDF3")
TEXT_SUB  = colors.HexColor("#8B949E")
BORDER    = colors.HexColor("#30363D")


def _emotion_colour(emotion: str) -> colors.HexColor:
    emap = {
        "Happy":    "#7EE787", "Sad":    "#58A6FF",
        "Angry":    "#F78166", "Fear":   "#D2A8FF",
        "Surprise": "#79C0FF", "Disgust":"#FFA657",
        "Neutral":  "#8B949E",
    }
    return colors.HexColor(emap.get(emotion, "#8B949E"))


def generate_pdf(report_data: Dict[str, Any], output_path: str) -> str:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=24, textColor=ACCENT, spaceAfter=4,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=10, textColor=TEXT_SUB, spaceAfter=2,
        fontName="Helvetica", alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=13, textColor=ACCENT, spaceBefore=14, spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body2", parent=styles["Normal"],
        fontSize=10, textColor=TEXT_MAIN, spaceAfter=4,
        fontName="Helvetica", leading=16,
    )
    rec_style = ParagraphStyle(
        "Rec", parent=styles["Normal"],
        fontSize=9.5, textColor=TEXT_MAIN, spaceAfter=3,
        fontName="Helvetica", leftIndent=12, leading=14,
    )

    story: List = []
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("EmotionSense AI", title_style))
    story.append(Paragraph("Interview & Behaviour Analytics Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=12))

    candidate = report_data.get("candidate_name", "Candidate")
    date_str  = report_data.get("date", datetime.datetime.now().strftime("%d %B %Y, %H:%M"))
    duration  = report_data.get("duration_seconds", 0)
    dur_str   = f"{duration // 60}m {duration % 60}s"

    meta_data = [
        ["Candidate", candidate,   "Date",     date_str],
        ["Duration",  dur_str,     "Session",  report_data.get("session_id", "—")],
    ]
    meta_table = Table(meta_data, colWidths=[3.5 * cm, 7 * cm, 3.5 * cm, 5.5 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), ACCENT),
        ("TEXTCOLOR",   (2, 0), (2, -1), ACCENT),
        ("TEXTCOLOR",   (1, 0), (1, -1), TEXT_MAIN),
        ("TEXTCOLOR",   (3, 0), (3, -1), TEXT_MAIN),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [CARD_BG, DARK_BG]),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Performance Summary", section_style))

    agg = report_data.get("aggregated", {})
    avg_conf   = agg.get("avg_confidence",  0)
    avg_stress = agg.get("avg_stress",      0)
    avg_att    = agg.get("avg_attention",   0)
    avg_eye    = agg.get("avg_eye_contact", 0)
    stress_lvl = agg.get("stress_level", "Low")
    att_state  = agg.get("attention_state", "Focused")

    def score_color(val):
        if val >= 70: return ACCENT2
        if val >= 45: return colors.HexColor("#FFA657")
        return WARN

    kpi_data = [
        ["Metric", "Score", "Status"],
        ["Confidence",   f"{avg_conf}%",   "High" if avg_conf > 70 else "Medium" if avg_conf > 45 else "Low"],
        ["Stress Level", f"{avg_stress}%",  stress_lvl],
        ["Attention",    f"{avg_att}%",     att_state],
        ["Eye Contact",  f"{avg_eye}%",     "Good" if avg_eye > 65 else "Needs Work"],
    ]
    kpi_table = Table(kpi_data, colWidths=[7 * cm, 4 * cm, 8.5 * cm])
    kpi_table.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("BACKGROUND",     (0, 0), (-1,  0), CARD_BG),
        ("TEXTCOLOR",      (0, 0), (-1,  0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK_BG, CARD_BG]),
        ("TEXTCOLOR",      (0, 1), ( 0, -1), TEXT_MAIN),
        ("TEXTCOLOR",      (1, 1), ( 1, -1), ACCENT2),
        ("TEXTCOLOR",      (2, 1), ( 2, -1), TEXT_MAIN),
        ("ALIGN",          (1, 0), (2, -1), "CENTER"),
        ("TOPPADDING",     (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
        ("LEFTPADDING",    (0, 0), (-1, -1), 10),
        ("LINEBELOW",      (0, 0), (-1,  0), 1, BORDER),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Emotion Distribution", section_style))
    emo_dist = agg.get("emotion_distribution", {})
    if emo_dist:
        sorted_emotions = sorted(emo_dist.items(), key=lambda x: x[1], reverse=True)
        emo_rows = [["Emotion", "Percentage", "Bar"]]
        for emo, pct in sorted_emotions:
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            emo_rows.append([emo, f"{pct}%", bar[:40]])
        emo_table = Table(emo_rows, colWidths=[4 * cm, 3 * cm, 12.5 * cm])
        emo_table.setStyle(TableStyle([
            ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("BACKGROUND",     (0, 0), (-1,  0), CARD_BG),
            ("TEXTCOLOR",      (0, 0), (-1,  0), ACCENT),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK_BG, CARD_BG]),
            ("TEXTCOLOR",      (0, 1), ( 0, -1), TEXT_MAIN),
            ("TEXTCOLOR",      (1, 1), ( 1, -1), ACCENT2),
            ("TEXTCOLOR",      (2, 1), ( 2, -1), ACCENT),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
            ("LINEBELOW",      (0, 0), (-1, 0), 1, BORDER),
        ]))
        story.append(emo_table)

    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("AI Recommendations", section_style))
    recs = agg.get("recommendations", [])
    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f"  {i}.  {rec}", rec_style))

    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 0.2 * cm))
    footer_txt = (
        f"Generated by EmotionSense AI  ·  {datetime.datetime.now().strftime('%d %b %Y %H:%M')}  "
        "·  Powered by MediaPipe + OpenCV"
    )
    story.append(Paragraph(footer_txt, ParagraphStyle(
        "Footer", fontSize=8, textColor=TEXT_SUB,
        fontName="Helvetica", alignment=TA_CENTER
    )))

    doc.build(story)
    return output_path
