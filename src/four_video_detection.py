import os
import sys
import csv
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# 0. PATH SETUP
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == "src"
    else SCRIPT_DIR
)

for p in (SCRIPT_DIR, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.append(p)

from traffic_controller import TrafficSignalController
from signal_control import SignalManager
from generative_ai import TrafficReportGenerator


# ============================================================
# 1. SETTINGS
# ============================================================

VIDEO_DIR = os.path.join(PROJECT_ROOT, "dataset", "videos")

VIDEO_FILES = {
    "Route 1": "road1.mp4",
    "Route 2": "road2.mp4",
    "Route 3": "road3.mp4",
    "Route 4": "road4.mp4",
}

VEHICLE_MODEL_NAME = "yolo11s.pt"     # CPU slow ayite yolo11n.pt pettandi
_local = os.path.join(PROJECT_ROOT, VEHICLE_MODEL_NAME)
VEHICLE_MODEL_PATH = _local if os.path.exists(_local) else VEHICLE_MODEL_NAME

AMBULANCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "vehicle_ambulance_best.pt"
)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "output_4videos.mp4")
OUT_DIR = os.path.dirname(OUTPUT_PATH)

# COCO: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]
SHORT_NAMES = {"car": "car", "motorcycle": "bike", "bus": "bus", "truck": "truck"}

VEHICLE_CONF = 0.20
AMBULANCE_CONF = 0.60     # ambulance modati nunchi detect avvakapothe 0.45 pettandi
IMGSZ = 640               # chinna vehicles miss ayite 960 pettandi
INTERVAL_SEC = 5          # vehicle count averaging window (video seconds)
FRAME_SKIP = 2            # slow ayite 3 leda 4
MAX_SECONDS = 180         # idi taruvata auto stop (None = Q varaku)

AMB_WINDOW = 4            # ambulance debounce: last 4 loops lo
AMB_HITS = 2              # kanisam 2 sarlu detect ayithe confirm
SIGNAL_TIME_SCALE = 1.0   # demo fast cheyyalante 2.0 (timers 2x speed)

TILE_W, TILE_H = 640, 360
HEADER_H = 95
BANNER_H = 64

# ---------- COLORS (BGR order) ----------
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

WAIT_COLORS = [
    (0, 0, 255),          # red     (next turn)
    (200, 60, 170),       # purple
    (0, 140, 255),        # orange
]


# ============================================================
# 2. CHECK VIDEOS
# ============================================================

ROUTES = list(VIDEO_FILES.keys())

missing = [
    f for f in VIDEO_FILES.values()
    if not os.path.exists(os.path.join(VIDEO_DIR, f))
]
if missing:
    print("ERROR: ee videos dorakaledu (dataset/videos lo pettandi):")
    for m in missing:
        print("  -", m)
    sys.exit()


# ============================================================
# 3. LOAD MODELS + CONTROLLERS
# ============================================================

vehicle_model = YOLO(VEHICLE_MODEL_PATH)

ambulance_model = None
if os.path.exists(AMBULANCE_MODEL_PATH):
    ambulance_model = YOLO(AMBULANCE_MODEL_PATH)
else:
    print("WARNING: Ambulance model dorakaledu:", AMBULANCE_MODEL_PATH)

controller = TrafficSignalController()
manager = SignalManager(ROUTES, controller)
reporter = TrafficReportGenerator()
decision_log = []


# ============================================================
# 4. OPEN VIDEOS
# ============================================================

caps = {}
for route, fname in VIDEO_FILES.items():
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, fname))
    if not cap.isOpened():
        print("ERROR: video open avvaledu:", fname)
        sys.exit()
    caps[route] = cap

fps_ref = caps[ROUTES[0]].get(cv2.CAP_PROP_FPS) or 25

os.makedirs(OUT_DIR, exist_ok=True)
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    max(1.0, fps_ref / FRAME_SKIP),
    (TILE_W * 2, TILE_H * 2),
)


# ============================================================
# 5. HELPERS
# ============================================================

def read_frame(cap):
    for _ in range(FRAME_SKIP - 1):
        cap.grab()
    ret, frame = cap.read()
    if not ret:                       # video aipothe malli modati nunchi
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    if not ret:
        return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    return cv2.resize(frame, (TILE_W, TILE_H))


def is_same_object(veh, amb):
    """COCO vehicle box, ambulance box tho overlap ayithe adi ambulance e."""
    ix1, iy1 = max(veh[0], amb[0]), max(veh[1], amb[1])
    ix2, iy2 = min(veh[2], amb[2]), min(veh[3], amb[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    va = max(1, (veh[2] - veh[0]) * (veh[3] - veh[1]))
    aa = max(1, (amb[2] - amb[0]) * (amb[3] - amb[1]))
    iou = inter / (va + aa - inter)
    return iou > 0.45 or inter / va > 0.75


def build_route_colors(avg):
    colors = {}
    cur = manager.current

    if manager.state == "GREEN":
        colors[cur] = GREEN
    elif manager.state == "YELLOW":
        colors[cur] = YELLOW

    waiting = manager.upcoming(avg)
    if manager.state not in ("GREEN", "YELLOW"):
        waiting = waiting + [cur]

    for i, r in enumerate(waiting):
        colors[r] = WAIT_COLORS[min(i, len(WAIT_COLORS) - 1)]
    return colors


def time_text(route, d, sig_t):
    if route == manager.current and manager.state == "GREEN":
        rem = manager.remaining(sig_t)
        if rem > 0:
            return f"{rem:.0f}/{int(manager.duration)}s"
        return f"EXT {sig_t - manager.phase_start:.0f}s"
    if route == manager.current and manager.state == "YELLOW":
        return f"{manager.remaining(sig_t):.0f}s"
    return f"{d['green_time']} sec"


# ============================================================
# 6. STATE
# ============================================================

hist_len = max(1, int(INTERVAL_SEC * fps_ref / FRAME_SKIP))
history = {r: deque(maxlen=hist_len) for r in ROUTES}
amb_hist = {r: deque(maxlen=AMB_WINDOW) for r in ROUTES}

simulated_amb = set()
green_log = {r: 0 for r in ROUTES}
peak_counts = {r: 0 for r in ROUTES}
loop_i = 0


# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:

    loop_i += 1
    video_time = loop_i * FRAME_SKIP / fps_ref

    if MAX_SECONDS and video_time > MAX_SECONDS:
        break

    sig_t = video_time * SIGNAL_TIME_SCALE

    frames = [read_frame(caps[r]) for r in ROUTES]

    # ---------- detection (4 frames okesari batch) ----------
    veh_results = vehicle_model.predict(
        frames,
        conf=VEHICLE_CONF,
        imgsz=IMGSZ,
        classes=VEHICLE_CLASSES,
        verbose=False,
    )

    if ambulance_model is not None:
        amb_results = ambulance_model.predict(
            frames, conf=AMBULANCE_CONF, imgsz=640, verbose=False
        )
    else:
        amb_results = [None] * len(ROUTES)

    counts = {}
    class_counts = {}
    veh_boxes = {}
    amb_boxes = {}

    for route, vres, ares in zip(ROUTES, veh_results, amb_results):

        amb_list = []
        if ares is not None:
            for box in ares.boxes:
                if ambulance_model.names[int(box.cls[0])] != "ambulance":
                    continue
                amb_list.append(tuple(map(int, box.xyxy[0])))
        amb_boxes[route] = amb_list

        boxes = []
        cc = {"car": 0, "bike": 0, "bus": 0, "truck": 0}

        for b in vres.boxes:
            bb = tuple(map(int, b.xyxy[0]))
            if any(is_same_object(bb, ab) for ab in amb_list):
                continue                      # double box (ambulance) skip
            boxes.append(bb)
            short = SHORT_NAMES.get(vehicle_model.names[int(b.cls[0])])
            if short:
                cc[short] += 1

        cc["amb"] = len(amb_list)
        veh_boxes[route] = boxes
        class_counts[route] = cc
        counts[route] = len(boxes) + len(amb_list)

        history[route].append(counts[route])
        amb_hist[route].append(1 if amb_list else 0)
        peak_counts[route] = max(peak_counts[route], counts[route])

    avg = {r: int(round(sum(h) / len(h))) for r, h in history.items()}

    amb_confirmed = {
        r for r in ROUTES if sum(amb_hist[r]) >= AMB_HITS
    } | simulated_amb
    amb_seen = {
        r: (len(amb_boxes[r]) > 0 or r in simulated_amb) for r in ROUTES
    }

    # ---------- signal state machine ----------
    event = manager.update(sig_t, avg, amb_confirmed, amb_seen)
    decisions = manager.snapshot(avg, amb_confirmed)
    selected = manager.current
    route_colors = build_route_colors(avg)

    if event is not None:
        green_log[selected] += 1
        decision_log.append(
            {
                "time": round(video_time, 1),
                "counts": dict(avg),
                "selected": selected,
                "green_time": event["green_time"],
                "ambulance": sorted(amb_confirmed),
                "reason": event["reason"],
            }
        )
        reporter.submit(
            video_time, dict(avg), selected, decisions,
            sorted(amb_confirmed), event["reason"],
        )
        print(
            f"[{video_time:5.1f}s] GREEN -> {selected} {event['green_time']} sec | "
            f"ambulance={sorted(amb_confirmed)} | counts={avg} | {event['reason']}"
        )

    # ---------- draw boxes ----------
    for route, frame in zip(ROUTES, frames):

        for (x1, y1, x2, y2) in veh_boxes[route]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), route_colors[route], 2)

        for (x1, y1, x2, y2) in amb_boxes[route]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 4)
            ly = max(y1 - 8, HEADER_H + 22)       # header tho overlap avvakunda
            cv2.putText(
                frame, "AMBULANCE", (x1, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2,
            )

    # ---------- draw tiles ----------
    tiles = []
    for route, frame in zip(ROUTES, frames):

        d = decisions[route]
        color = route_colors[route]
        status = {"GREEN": "GREEN", "YELLOW": "YELLOW"}.get(d["signal"], "WAIT")
        cc = class_counts[route]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (TILE_W, HEADER_H), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Line 1: route, status, countdown / green time
        title = f"{route} [{status}] {time_text(route, d, sig_t)}"
        if d["ambulance"]:
            title += "  AMBULANCE"
        cv2.putText(
            frame, title, (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

        # Line 2: vehicle count
        cv2.putText(
            frame, f"VEHICLES: {counts[route]}", (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHITE, 2,
        )
        cv2.putText(
            frame, f"avg {d['vehicles']} | {d['density']}", (300, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2,
        )

        # Line 3: breakdown
        breakdown = (
            f"car {cc['car']} | bike {cc['bike']} | bus {cc['bus']} | "
            f"truck {cc['truck']} | amb {cc['amb']}"
        )
        cv2.putText(
            frame, breakdown, (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1,
        )

        # Countdown progress bar (active green route ki)
        if route == manager.current and manager.state == "GREEN" and manager.duration > 0:
            frac = min(1.0, manager.remaining(sig_t) / manager.duration)
            cv2.rectangle(
                frame, (0, HEADER_H - 6), (int(TILE_W * frac), HEADER_H), color, -1
            )

        cv2.rectangle(frame, (0, 0), (TILE_W - 1, TILE_H - 1), color, 5)
        tiles.append(frame)

    canvas = np.vstack(
        [np.hstack(tiles[0:2]), np.hstack(tiles[2:4])]
    )

    # ---------- bottom banner ----------
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (0, TILE_H * 2 - BANNER_H),
        (TILE_W * 2, TILE_H * 2),
        (0, 0, 0),
        -1,
    )
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    state_txt = {"GREEN": "GREEN", "YELLOW": "YELLOW", "ALL_RED": "ALL-RED"}.get(
        manager.state, manager.state
    )
    banner = (
        f"{selected or 'Starting'}: {state_txt} {manager.remaining(sig_t):.0f}s left | "
        f"TOTAL VEHICLES: {sum(counts.values())}"
    )
    if manager.amb_phase and manager.state == "GREEN":
        banner += " | AMBULANCE PRIORITY"
        if selected in simulated_amb:
            banner += " (SIM)"
    cv2.putText(
        canvas, banner, (15, TILE_H * 2 - 11),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2,
    )

    ai_text = reporter.latest.encode("ascii", "ignore").decode()[:120]
    cv2.putText(
        canvas, "AI: " + ai_text, (15, TILE_H * 2 - 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
    )

    writer.write(canvas)
    cv2.imshow("Smart Traffic Signal System (4 Routes)", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
        simulated_amb = {f"Route {chr(key)}"}
    elif key == ord("0"):
        simulated_amb = set()


# ============================================================
# 8. RELEASE + SUMMARY + REPORT
# ============================================================

for cap in caps.values():
    cap.release()
writer.release()
cv2.destroyAllWindows()

print("\n===== SUMMARY =====")
for r in ROUTES:
    print(
        f"{r}: green phases {green_log[r]} | "
        f"peak vehicles in a frame: {peak_counts[r]}"
    )
print("Output saved:", OUTPUT_PATH)

if decision_log:
    csv_path = os.path.join(OUT_DIR, "decision_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["time_s", "selected_route", "green_time_s", "ambulance", "reason"]
            + [f"{r}_vehicles" for r in ROUTES]
        )
        for x in decision_log:
            w.writerow(
                [x["time"], x["selected"], x["green_time"],
                 ";".join(x["ambulance"]), x["reason"]]
                + [x["counts"][r] for r in ROUTES]
            )
    print("CSV saved:", csv_path)

    report_path = reporter.save_report(
        decision_log, peak_counts, green_log, OUT_DIR
    )
    print("Report saved:", report_path)