
import os
import cv2
from ultralytics import YOLO

# ============================================================
# 1. PROJECT PATHS - MATCHED TO YOUR VS CODE STRUCTURE
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(
    PROJECT_ROOT, "main.py", "model", "models"
)

# Try your available vehicle models in this order.
vehicle_candidates = [
    os.path.join(MODELS_DIR, "vehicle_model_trained.pt"),
    os.path.join(MODELS_DIR, "vehicle_model.pt"),
    os.path.join(MODELS_DIR, "best.pt"),
]

VEHICLE_MODEL_PATH = next(
    (path for path in vehicle_candidates if os.path.isfile(path)),
    None
)

AMBULANCE_MODEL_PATH = os.path.join(
    MODELS_DIR, "vehicle_ambulance_best.pt"
)

VIDEO_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "videos", "traffic.mp4"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.mp4")
TRACKER_PATH = os.path.join(OUTPUT_DIR, "bytetrack.yaml")

# ============================================================
# 2. SETTINGS
# ============================================================

FRAME_W, FRAME_H = 1280, 720
FRAME_SKIP = 2
INTERVAL_SEC = 5
VEHICLE_CONF = 0.25
AMBULANCE_CONF = 0.60
IMGSZ = 640

LANES = ["Lane 1", "Lane 2", "Lane 3"]

# Divider x-coordinates at top and bottom of video.
LANE_DIVIDERS = [
    (430, 430),
    (850, 850),
]

LANE_COLORS = {
    "Lane 1": (255, 160, 0),
    "Lane 2": (0, 220, 0),
    "Lane 3": (0, 160, 255),
}

# ============================================================
# 3. CHECK PATHS AND LOAD MODELS
# ============================================================

if VEHICLE_MODEL_PATH is None:
    raise FileNotFoundError(
        "Vehicle model not found in:\n"
        + MODELS_DIR
        + "\nExpected vehicle_model_trained.pt or vehicle_model.pt"
    )

if not os.path.isfile(AMBULANCE_MODEL_PATH):
    raise FileNotFoundError(
        "Ambulance model not found:\n" + AMBULANCE_MODEL_PATH
    )

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(
        "Video not found:\n" + VIDEO_PATH
    )

print("Vehicle model:", VEHICLE_MODEL_PATH)
print("Ambulance model:", AMBULANCE_MODEL_PATH)
print("Input video:", VIDEO_PATH)

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
ambulance_model = YOLO(AMBULANCE_MODEL_PATH)

print("Vehicle classes:", vehicle_model.names)
print("Ambulance classes:", ambulance_model.names)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TRACKER_PATH, "w", encoding="utf-8") as f:
    f.write(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.10\n"
        "new_track_thresh: 0.25\n"
        "track_buffer: 30\n"
        "match_thresh: 0.80\n"
        "fuse_score: True\n"
    )

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def divider_x(divider, y):
    ratio = max(0, min(1, y / FRAME_H))
    top, bottom = divider
    return top + (bottom - top) * ratio


def get_lane(cx, cy):
    if cx < divider_x(LANE_DIVIDERS[0], cy):
        return "Lane 1"
    if cx < divider_x(LANE_DIVIDERS[1], cy):
        return "Lane 2"
    return "Lane 3"


def green_seconds(count):
    if count <= 5:
        return 20
    if count <= 10:
        return 30
    if count <= 20:
        return 45
    return 60


def is_vehicle(label):
    # Works with common vehicle class names from custom models.
    name = str(label).lower().replace("_", " ").strip()
    vehicle_words = (
        "car", "bus", "truck", "motorcycle",
        "motorbike", "bike", "van", "vehicle"
    )
    return any(word in name for word in vehicle_words)


def is_ambulance(label):
    name = str(label).lower().strip()
    return "ambulance" in name or "ambulence" in name


def put_text(frame, text, position, color=(255, 255, 255),
             scale=0.6, thickness=2):
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA
    )


# ============================================================
# 5. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Cannot open video: " + VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    max(1.0, fps / FRAME_SKIP),
    (FRAME_W, FRAME_H),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video: " + OUTPUT_PATH)

# ============================================================
# 6. TRACKING AND SIGNAL STATE
# ============================================================

frame_number = 0
last_decision_time = 0.0

# Track ID -> lane, within the current interval.
interval_vehicle_ids = {}
interval_ambulance_lanes = set()

selected_lane = None
green_duration = 0
reason = "Waiting for detections"

# ============================================================
# 7. PROCESS VIDEO
# ============================================================

try:
    while True:
        ret, raw_frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number % FRAME_SKIP != 0:
            continue

        video_time = frame_number / fps
        frame = cv2.resize(raw_frame, (FRAME_W, FRAME_H))
        clean_frame = frame.copy()

        live_counts = {lane: 0 for lane in LANES}

        # ----------------------------------------------------
        # A. VEHICLE DETECTION AND TRACKING
        # ----------------------------------------------------

        vehicle_results = vehicle_model.track(
            clean_frame,
            persist=True,
            tracker=TRACKER_PATH,
            conf=VEHICLE_CONF,
            imgsz=IMGSZ,
            verbose=False,
        )

        for result in vehicle_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = vehicle_model.names[class_id]

                if not is_vehicle(label):
                    continue

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                lane = get_lane(cx, cy)

                live_counts[lane] += 1

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())
                    interval_vehicle_ids[track_id] = lane

                display_label = (
                    f"{label} ID:{track_id}"
                    if track_id is not None else str(label)
                )

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    LANE_COLORS[lane], 2
                )
                put_text(
                    frame, display_label,
                    (x1, max(20, y1 - 8)),
                    (0, 255, 255), 0.5, 1
                )

        # Combine current detections and unique tracked IDs.
        unique_counts = {lane: 0 for lane in LANES}
        for lane in interval_vehicle_ids.values():
            unique_counts[lane] += 1

        counts = {
            lane: max(live_counts[lane], unique_counts[lane])
            for lane in LANES
        }

        # ----------------------------------------------------
        # B. AMBULANCE DETECTION
        # ----------------------------------------------------

        ambulance_results = ambulance_model.predict(
            clean_frame,
            conf=AMBULANCE_CONF,
            imgsz=640,
            verbose=False,
        )

        current_ambulance_lanes = set()

        for result in ambulance_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = ambulance_model.names[class_id]

                if not is_ambulance(label):
                    continue

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                lane = get_lane(cx, cy)

                current_ambulance_lanes.add(lane)
                interval_ambulance_lanes.add(lane)

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 0, 255), 3
                )
                put_text(
                    frame, "AMBULANCE",
                    (x1, max(25, y1 - 8)),
                    (0, 0, 255), 0.7, 2
                )

        # ----------------------------------------------------
        # C. SELECT GREEN LANE EVERY 5 VIDEO SECONDS
        # ----------------------------------------------------

        if video_time - last_decision_time >= INTERVAL_SEC:
            if interval_ambulance_lanes:
                # Deterministic priority if multiple lanes detect
                # an ambulance. A real system needs arbitration.
                selected_lane = sorted(
                    interval_ambulance_lanes
                )[0]
                green_duration = 30
                reason = "AMBULANCE PRIORITY"

            else:
                selected_lane = max(
                    LANES, key=lambda lane: counts[lane]
                )
                green_duration = green_seconds(
                    counts[selected_lane]
                )
                reason = "DENSITY BASED"

            print(
                f"Time {video_time:.1f}s | Counts: {counts} | "
                f"Green: {selected_lane} | "
                f"Duration: {green_duration}s | {reason}"
            )

            last_decision_time = video_time
            interval_vehicle_ids.clear()
            interval_ambulance_lanes.clear()

        # ----------------------------------------------------
        # D. DRAW LANE COUNTS AND SIGNALS
        # ----------------------------------------------------

        for index, lane in enumerate(LANES):
            put_text(
                frame,
                f"{lane}: {counts[lane]} vehicles",
                (20, 35 + index * 30),
                LANE_COLORS[lane], 0.65, 2
            )

        put_text(
            frame, f"Time: {video_time:.1f}s",
            (20, 135)
        )
        put_text(
            frame, f"Decision: {reason}",
            (20, 170), (0, 255, 255)
        )
        put_text(
            frame, f"Green duration: {green_duration}s",
            (20, 205)
        )

        # Draw exactly one green lane in this simulation.
        for index, lane in enumerate(LANES):
            x, y = FRAME_W - 200, 45 + index * 95
            is_green = lane == selected_lane

            color = (0, 180, 0) if is_green else (0, 0, 220)
            status = "GREEN" if is_green else "RED"

            cv2.rectangle(
                frame, (x, y), (x + 175, y + 65),
                color, -1
            )
            put_text(
                frame, f"{lane}: {status}",
                (x + 8, y + 38), (255, 255, 255), 0.55, 2
            )

        for divider in LANE_DIVIDERS:
            x_top, x_bottom = divider
            cv2.line(
                frame, (x_top, 0), (x_bottom, FRAME_H),
                (255, 255, 255), 2
            )

        writer.write(frame)

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("\nVideo processing completed.")
print("Output video:", OUTPUT_PATH)
