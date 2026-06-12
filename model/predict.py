import cv2, math, time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_CD     = cv2.data.haarcascades
_FACE   = cv2.CascadeClassifier(_CD + "haarcascade_frontalface_default.xml")
_FACE2  = cv2.CascadeClassifier(_CD + "haarcascade_frontalface_alt2.xml")
_EYE    = cv2.CascadeClassifier(_CD + "haarcascade_eye.xml")
_SMILE  = cv2.CascadeClassifier(_CD + "haarcascade_smile.xml")


@dataclass
class EmotionResult:
    dominant:          str             = "Neutral"
    scores:            Dict[str,float] = field(default_factory=dict)
    confidence_score:  float           = 0.0
    stress_level:      str             = "Low"
    stress_score:      float           = 0.0
    attention_state:   str             = "Focused"
    attention_score:   float           = 0.0
    eye_contact_score: float           = 0.0
    head_stability:    float           = 0.0
    smile_score:       float           = 0.0
    fps:               float           = 0.0
    face_found:        bool            = False


class FaceTracker:
    EMOTIONS     = ["Happy","Sad","Angry","Fear","Surprise","Disgust","Neutral"]
    CALIB_FRAMES = 60          # keep a neutral face during this window
    EMA_ALPHA    = 0.30        # smoothing speed  (0=frozen, 1=instant)

    _CAL_KEYS = ["smile_conf","eye_open","brow_h","mouth_h_n",
                 "mouth_w_n","brow_gap_n","up_bright","lo_bright"]

    def __init__(self):
        self._cal    : Dict[str,list]  = {}
        self._base   : Dict[str,float] = {}
        self._cal_n  : int   = 0
        self._ready  : bool  = False
        self._smooth : Dict[str,float] = {e: 0.0 for e in self.EMOTIONS}
        self._prev   : Optional[Tuple[float,float]] = None
        self._moves  : deque = deque(maxlen=30)
        self._ec_buf : deque = deque(maxlen=8)

    def update_calibration(self, f: Dict[str,float]):
        for k in self._CAL_KEYS:
            self._cal.setdefault(k, []).append(f.get(k, 0.0))
        self._cal_n += 1
        if self._cal_n >= self.CALIB_FRAMES and not self._ready:
            for k in self._CAL_KEYS:
                self._base[k] = float(np.median(self._cal[k]))
            self._ready = True

    def delta(self, f: Dict[str,float], k: str) -> float:
        return (f[k] - self._base.get(k, f[k])) if self._ready else 0.0

    @property
    def calibrated(self) -> bool: return self._ready
    @property
    def calib_pct(self) -> float: return min(1.0, self._cal_n / self.CALIB_FRAMES)

    def score_emotions(self, f: Dict[str,float]) -> Dict[str,float]:
        if not self._ready:
            return {e: (100.0 if e == "Neutral" else 0.0) for e in self.EMOTIONS}

        d_smile = self.delta(f, "smile_conf")   
        d_eye   = self.delta(f, "eye_open")    
        d_brow  = self.delta(f, "brow_h")       
        d_mth_h = self.delta(f, "mouth_h_n")   
        d_mth_w = self.delta(f, "mouth_w_n")   
        d_bgap  = self.delta(f, "brow_gap_n")  
        d_upbr  = self.delta(f, "up_bright")   

        def mx(v): return max(0.0, float(v))

        squint  = mx(-d_eye)    # eyes squinting (angry)
        wide    = mx( d_eye)    # eyes wide open (fear/surprise)
        gap_neg = mx(-d_bgap)   # brows drawn together (fear/angry)
        gap_pos = mx( d_bgap)   # brows spread apart (surprise)
        brow_up = mx( d_brow)   # brows raised (fear/surprise)
        brow_dn = mx(-d_brow)   # brows lowered (sad/angry)

        happy    = mx(d_smile * 20.0) + mx(d_eye * 1.5)

        sad      = max(0.0,
                       mx(brow_dn * 20.0)
                       + mx(-d_smile * 3.0)
                       - squint * 14.0)

        angry    = mx(squint * 9.0) + mx(gap_neg * 9.0) + mx(brow_dn * 1.5)

        fear     = (brow_up * gap_neg * 300.0
                    + mx(wide * 4.0)
                    + mx(d_mth_h * 2.0))

        surprise = (brow_up * gap_pos * 300.0
                    + mx(wide * 4.0)
                    + mx(d_mth_h * 15.0))

        no_strong = max(0.0, 0.10 - squint * 0.5 - gap_neg * 0.5)
        disgust  = (mx(-d_smile * 16.0)
                    + mx(-d_mth_w * 8.0)
                    + no_strong * 4.0
                    + mx(d_upbr * 3.0))

        total_delta = (abs(d_smile) + abs(d_eye) + abs(d_brow)
                       + abs(d_mth_h) + abs(d_mth_w) + abs(d_bgap))
        neutral  = max(0.0, 0.65 - total_delta * 5.0)

        raw  = {"Happy": happy, "Sad": sad, "Angry": angry,
                "Fear": fear,  "Surprise": surprise,
                "Disgust": disgust, "Neutral": neutral}
        vals = np.array([raw[e] for e in self.EMOTIONS], np.float64)
        vals -= vals.min()
        ev   = np.exp(vals / 0.28)
        sm   = ev / ev.sum()
        return {e: round(float(sm[i]) * 100, 1) for i, e in enumerate(self.EMOTIONS)}

    def smooth(self, sc: Dict[str,float]) -> Dict[str,float]:
        a = self.EMA_ALPHA
        for e in self.EMOTIONS:
            self._smooth[e] = a * sc[e] + (1 - a) * self._smooth[e]
        tot = sum(self._smooth.values()) + 1e-8
        return {e: round(self._smooth[e] / tot * 100, 1) for e in self.EMOTIONS}

    def head_stability(self, cx: float, cy: float, fw: float) -> float:
        if self._prev is not None:
            dx = (cx - self._prev[0]) / (fw + 1e-4)
            dy = (cy - self._prev[1]) / (fw + 1e-4)
            self._moves.append(math.sqrt(dx * dx + dy * dy))
        self._prev = (cx, cy)
        if not self._moves:
            return 100.0
        avg = float(np.mean(self._moves))
        return round(float(np.clip(1.0 - avg * 35.0, 0.0, 1.0)) * 100.0, 1)

    def eye_contact(self, eyes_n: int, yaw_est: float) -> float:
        # eyes_n=2 → looking at camera; 1 → turned slightly; 0 → turned away
        eye_score = 1.0 if eyes_n >= 2 else 0.60 if eyes_n == 1 else 0.20
        yaw_score = float(np.clip(1.0 - abs(yaw_est) / 35.0, 0.0, 1.0))
        ec = (eye_score * 0.65 + yaw_score * 0.35) * 100.0
        self._ec_buf.append(ec)
        return round(float(np.mean(self._ec_buf)), 1)


class EmotionDetector:
    EMOTIONS = FaceTracker.EMOTIONS

    def __init__(self):
        self._trackers  : Dict[int, FaceTracker]        = {}
        self._last_faces: List[Tuple[int,int,int]]       = []  # (id, cx, cy)
        self._fps_buf   : deque                          = deque(maxlen=30)

    @staticmethod
    def _detect_faces(gray: np.ndarray) -> List[Tuple]:
        kw = dict(scaleFactor=1.08, minNeighbors=5,
                  minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)
        faces = _FACE.detectMultiScale(gray, **kw)
        if len(faces) == 0:
            faces = _FACE2.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=4, minSize=(80, 80))
        return [tuple(map(int, f)) for f in faces] if len(faces) else []

    @staticmethod
    def _extract(gray: np.ndarray, frame: np.ndarray,
                 fx: int, fy: int, fw: int, fh: int) -> Dict[str,float]:

        fg  = gray[fy:fy+fh, fx:fx+fw]
        fc  = frame[fy:fy+fh, fx:fx+fw]
        mid = fw // 2

        brow_roi  = fg[0           : int(fh*.30), :]
        eye_roi   = fg[int(fh*.20) : int(fh*.48), :]
        mouth_roi = fg[int(fh*.58) : int(fh*.90),
                       int(fw*.08) : int(fw*.92)]
        upper_col = fc[0           : int(fh*.48), :]
        lower_col = fc[int(fh*.48) :,              :]

        smile_conf = 0.0
        smiles = _SMILE.detectMultiScale(
            mouth_roi, scaleFactor=1.6, minNeighbors=22,
            minSize=(int(fw*.15), int(fh*.05)))
        if len(smiles) > 0:
            smile_conf = min(1.0, len(smiles) * 0.40 + 0.30)
        else:
            smiles2 = _SMILE.detectMultiScale(
                mouth_roi, scaleFactor=1.5, minNeighbors=14,
                minSize=(int(fw*.10), int(fh*.04)))
            smile_conf = min(0.50, len(smiles2) * 0.20)

        eyes   = _EYE.detectMultiScale(
            eye_roi, scaleFactor=1.15, minNeighbors=4, minSize=(18, 18))
        eyes_n = int(len(eyes))
        if eyes_n > 0:
            avg_eye_h = float(np.mean([e[3] for e in eyes]))
            eye_open  = float(np.clip(avg_eye_h / (fh * 0.12), 0.2, 1.0))
        else:
            grad     = cv2.Laplacian(eye_roi, cv2.CV_64F)
            eye_open = float(np.clip(np.std(grad) / 18.0, 0.1, 1.0))

        if brow_roi.size > 0:
            sobel  = cv2.Sobel(brow_roi, cv2.CV_64F, 0, 1, ksize=3)
            brow_h = float(np.clip(
                np.mean(np.abs(sobel)) / (fw * 0.5 + 1e-4) * 15.0, 0.0, 1.0))
        else:
            brow_h = 0.4

        if brow_roi.size > 4 and fw > 6:
            c = brow_roi[:, fw//3 : 2*fw//3]
            s = np.concatenate(
                [brow_roi[:, :fw//4], brow_roi[:, 3*fw//4:]], axis=1)
            if s.size > 0 and c.size > 0:
                brow_gap_n = float(np.clip(
                    (float(np.mean(s)) - float(np.mean(c))) / 255.0 * 4.0,
                    -1.0, 1.0))
            else:
                brow_gap_n = 0.0
        else:
            brow_gap_n = 0.0

        if mouth_roi.size > 0:
            _, th     = cv2.threshold(mouth_roi, 55, 255, cv2.THRESH_BINARY_INV)
            rows_pct  = np.sum(th, axis=1) / (th.shape[1] * 255.0 + 1e-4)
            mouth_h_n = float(np.clip(
                np.sum(rows_pct > 0.25) / (fh * 0.25 + 1e-4), 0.0, 1.0))
        else:
            mouth_h_n = 0.0

        m_det     = _SMILE.detectMultiScale(
            mouth_roi, scaleFactor=1.5, minNeighbors=10,
            minSize=(int(fw*.08), int(fh*.03)))
        mouth_w_n = (float(np.mean([d[2] for d in m_det])) / fw
                     if len(m_det) > 0 else 0.40)

        up_br = float(np.mean(upper_col)) / 255.0 if upper_col.size > 0 else 0.5
        lo_br = float(np.mean(lower_col)) / 255.0 if lower_col.size > 0 else 0.5

        lh      = float(np.mean(fg[:, :mid])) if mid > 0 else 128.0
        rh      = float(np.mean(fg[:, mid:])) if mid > 0 else 128.0
        yaw_est = float(np.clip((lh - rh) / 30.0, -1.5, 1.5)) * 30.0

        return dict(
            smile_conf=smile_conf, eye_open=eye_open, eyes_n=eyes_n,
            brow_h=brow_h, mouth_h_n=mouth_h_n, mouth_w_n=mouth_w_n,
            brow_gap_n=brow_gap_n, up_bright=up_br, lo_bright=lo_br,
            yaw_est=yaw_est,
        )

    def _get_tracker(self, cx: int, cy: int) -> Tuple["FaceTracker", int]:
        best_id, best_d = None, 9999.0
        for tid, tcx, tcy in self._last_faces:
            d = math.sqrt((cx - tcx)**2 + (cy - tcy)**2)
            if d < best_d:
                best_d = d; best_id = tid
        if best_id is None or best_d > 120:
            new_id = (max(t for t,_,_ in self._last_faces)
                      if self._last_faces else -1) + 1
            self._trackers[new_id] = FaceTracker()
            return self._trackers[new_id], new_id
        if best_id not in self._trackers:
            self._trackers[best_id] = FaceTracker()
        return self._trackers[best_id], best_id

    @staticmethod
    def _draw(frame, fx, fy, fw, fh, dom, sc, calibrated, calib_pct):
        COLS = {
            "Happy"   : (80,  220, 120),
            "Sad"     : (200,  90,  50),
            "Angry"   : ( 60,  60, 220),
            "Fear"    : (190,  60, 200),
            "Surprise": ( 30, 210, 220),
            "Disgust" : ( 50, 180,  60),
            "Neutral" : (160, 160, 160),
        }
        col = COLS.get(dom, (160, 160, 160))

        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), col, 2)

        label      = f"{dom}  {sc.get(dom, 0):.0f}%"
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(frame,
                      (fx, fy - lh - 12), (fx + lw + 10, fy), col, -1)
        cv2.putText(frame, label, (fx + 5, fy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

        if not calibrated:
            cv2.rectangle(frame,
                          (fx, fy+fh+4), (fx+fw, fy+fh+11),
                          (40, 40, 40), -1)
            cv2.rectangle(frame,
                          (fx, fy+fh+4),
                          (fx + int(fw * calib_pct), fy+fh+11),
                          (80, 200, 120), -1)
            cv2.putText(frame,
                        f"Calibrating {int(calib_pct*100)}% — keep neutral",
                        (fx, fy+fh+24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 200, 120), 1)

    def process_frame(self, frame: np.ndarray):
        t0   = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # normalise illumination variation

        faces = self._detect_faces(gray)
        if not faces:
            self._fps_buf.append(time.time() - t0)
            return frame, [EmotionResult(face_found=False)]

        results   : List[EmotionResult]          = []
        new_faces : List[Tuple[int,int,int]]      = []

        for fx, fy, fw, fh in faces:
            cx, cy = fx + fw // 2, fy + fh // 2
            tracker, tid = self._get_tracker(cx, cy)
            new_faces.append((tid, cx, cy))

            feats = self._extract(gray, frame, fx, fy, fw, fh)
            tracker.update_calibration(feats)

            raw_sc = tracker.score_emotions(feats)
            sc     = tracker.smooth(raw_sc)
            dom    = max(sc, key=sc.get)

            stab = tracker.head_stability(float(cx), float(cy), float(fw))

            ec = tracker.eye_contact(feats["eyes_n"], feats["yaw_est"])

            s_raw  = (sc.get("Angry",0)*0.35 + sc.get("Fear",0)*0.30
                      + sc.get("Sad",0)*0.15 + max(0, 60-ec)*0.20)
            stress = round(float(np.clip(s_raw / 100.0 * 1.5, 0.0, 1.0)) * 100, 1)
            s_lvl  = "High" if stress > 60 else "Medium" if stress > 35 else "Low"

            eye_bonus = 20.0 if feats["eyes_n"] >= 2 else 0.0
            att   = round(float(np.clip(
                ec * 0.55 + stab * 0.30 + eye_bonus * 0.15, 0.0, 100.0)), 1)
            att_st = "Focused" if att > 58 else "Distracted"

            smile_bonus = float(np.clip(feats["smile_conf"] * 20.0, 0.0, 20.0))
            pos_emo     = sc.get("Happy", 0)*0.40 + sc.get("Neutral", 0)*0.30
            conf = round(float(np.clip(
                ec   * 0.35
                + stab * 0.30
                + pos_emo * 0.25
                + smile_bonus * 0.10,
                0.0, 100.0)), 1)

            smile_sc = round(feats["smile_conf"] * 100.0, 1)

            self._draw(frame, fx, fy, fw, fh, dom, sc,
                       tracker.calibrated, tracker.calib_pct)

            results.append(EmotionResult(
                dominant          = dom,
                scores            = sc,
                confidence_score  = conf,
                stress_level      = s_lvl,
                stress_score      = stress,
                attention_state   = att_st,
                attention_score   = att,
                eye_contact_score = ec,
                head_stability    = stab,
                smile_score       = smile_sc,
                face_found        = True,
            ))

        self._last_faces = new_faces
        self._fps_buf.append(time.time() - t0)
        fps = round(1.0 / (float(np.mean(self._fps_buf)) + 1e-8), 1)
        for r in results:
            r.fps = fps
        return frame, results
