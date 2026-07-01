import cv2
import os
import time
import csv


# ============================================================
# Project Title:
# User-Selected Object Tracking in a Multi-Object Scene - Safe Mode
#
# Objective:
# Track one user-selected object in a video containing multiple objects.
# If the target is lost or leaves the frame, the system displays
# TARGET LOST and does not automatically switch to another object.
#
# Controls:
# q / 1  -> quit
# s / 2  -> save current frame
# r / 3  -> reset target manually
# p / 4  -> pause / resume
# t      -> change tracker type
# ============================================================


# =========================
# Settings
# =========================

VIDEO_PATH = r"data\test2.mpg"
MAX_WIDTH = 900
OUTPUT_DIR = "output"
SHOW_DEBUG_WINDOWS = True
SHOW_TEXT = True

# ROI validation
MIN_ROI_WIDTH = 20
MIN_ROI_HEIGHT = 20
MIN_ROI_AREA = 500

# Lost detection
MAX_LOST_FRAMES = 5
MIN_VISIBLE_RATIO = 0.55

# Bounding box stabilization
STABILIZE_BOX_SIZE = True
MAX_GROWTH_FROM_INITIAL = 1.15
MAX_GROWTH_PER_FRAME = 1.04
MIN_BOX_SIZE = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Basic Utilities
# =========================

def resize_frame(frame, max_width=900):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


def preprocess_frame(frame):
    """
    Preprocessing pipeline: grayscale → → edges → contours.
    Only called when SHOW_DEBUG_WINDOWS is True.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_view = frame.copy()
    cv2.drawContours(contour_view, contours, -1, (0, 255, 255), 1)
    return edges, contour_view


def choose_tracker_type(current_tracker=None):
    print("\nChoose tracker type:")
    print("1 - CSRT  | More accurate, slower")
    print("2 - KCF   | Faster, medium accuracy")
    print("3 - MOSSE | Very fast, lower accuracy")

    if current_tracker:
        print(f"Press ENTER to keep current tracker: {current_tracker}")
    else:
        print("Press ENTER for default CSRT")

    choice = input("Your choice: ").strip()

    if choice == "2":
        return "KCF"
    if choice == "3":
        return "MOSSE"
    if current_tracker and choice == "":
        return current_tracker
    return "CSRT"


def create_tracker(tracker_name):
    """Create OpenCV tracker. Requires opencv-contrib-python."""
    trackers = {
        "CSRT":  cv2.legacy.TrackerCSRT_create,
        "KCF":   cv2.legacy.TrackerKCF_create,
        "MOSSE": cv2.legacy.TrackerMOSSE_create,
    }
    name = tracker_name.upper()
    if name not in trackers:
        raise ValueError(f"Unsupported tracker type: {tracker_name}")
    return trackers[name]()


# =========================
# ROI Selection
# =========================

def is_valid_roi(bbox):
    """Prevent tiny accidental selections."""
    if bbox is None:
        return False
    x, y, w, h = [int(v) for v in bbox]
    return w >= MIN_ROI_WIDTH and h >= MIN_ROI_HEIGHT and w * h >= MIN_ROI_AREA


def select_target(frame, allow_cancel=False):
    """Let user draw a bounding box. Re-prompts if selection is too small."""
    while True:
        cv2.namedWindow("Select Target Object", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Target Object", 1000, 650)
        bbox = cv2.selectROI("Select Target Object", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target Object")

        bbox = tuple(int(v) for v in bbox)

        if bbox == (0, 0, 0, 0):
            print("No ROI selected.")
            if allow_cancel:
                return None
            print("Please select the full target object.")
            continue

        if not is_valid_roi(bbox):
            print(
                f"Invalid ROI {bbox}. "
                f"Minimum size: {MIN_ROI_WIDTH}x{MIN_ROI_HEIGHT}, "
                f"minimum area: {MIN_ROI_AREA}."
            )
            if allow_cancel:
                return None
            print("Please select a larger box around the object.")
            continue

        return bbox


# =========================
# Bounding Box Utilities
# =========================

def bbox_visible_ratio(bbox, frame_w, frame_h):
    """How much of bbox is still inside the frame (0.0 – 1.0)."""
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return 0.0
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1) / (w * h)


def is_bbox_lost(bbox, frame_w, frame_h):
    """True if the target has mostly left the frame."""
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return True
    cx, cy = x + w / 2, y + h / 2
    if cx < 0 or cx > frame_w or cy < 0 or cy > frame_h:
        return True
    return bbox_visible_ratio(bbox, frame_w, frame_h) < MIN_VISIBLE_RATIO


def clamp_bbox(bbox, frame_w, frame_h):
    """Clamp bbox to frame boundaries. Returns None if fully outside."""
    x, y, w, h = [int(v) for v in bbox]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


# =========================
# Bounding Box Stabilization
# =========================

def stabilize_bbox_size(candidate, last, initial):
    """Keep tracker position but prevent bbox size from drifting."""
    if candidate is None:
        return None

    cx = candidate[0] + candidate[2] / 2
    cy = candidate[1] + candidate[3] / 2

    def safe(v):
        return max(MIN_BOX_SIZE, float(v))

    cw, ch = safe(candidate[2]), safe(candidate[3])
    lw, lh = safe(last[2]),      safe(last[3])
    iw, ih = safe(initial[2]),   safe(initial[3])

    max_w = min(iw * MAX_GROWTH_FROM_INITIAL, lw * MAX_GROWTH_PER_FRAME)
    max_h = min(ih * MAX_GROWTH_FROM_INITIAL, lh * MAX_GROWTH_PER_FRAME)

    new_w = min(cw, max_w)
    new_h = min(ch, max_h)

    # Smooth size changes: slower growth, faster shrink
    alpha_w = 0.15 if new_w > lw else 0.40
    alpha_h = 0.15 if new_h > lh else 0.40

    new_w = max(MIN_BOX_SIZE, (1 - alpha_w) * lw + alpha_w * new_w)
    new_h = max(MIN_BOX_SIZE, (1 - alpha_h) * lh + alpha_h * new_h)

    return int(cx - new_w / 2), int(cy - new_h / 2), int(new_w), int(new_h)


# =========================
# Output
# =========================

def draw_output(frame, bbox, status, fps, frame_number, tracker_name):
    output = frame.copy()
    frame_h, frame_w = output.shape[:2]

    color = {
        "TRACKING":    (0, 255, 0),
        "WARNING":     (0, 255, 255),
        "TARGET LOST": (0, 0, 255),
    }.get(status, (0, 0, 255))

    # Draw bounding box (bbox is None when target is lost)
    if bbox is not None:
        safe = clamp_bbox(bbox, frame_w, frame_h)
        if safe is not None:
            x, y, w, h = safe
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output, "TARGET", (x, max(15, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    if SHOW_TEXT:
        cv2.putText(
            output,
            f"{status} | Tracker:{tracker_name} | FPS:{fps:.1f} | Frame:{frame_number}",
            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA
        )
        cv2.putText(
            output,
            "q/1 quit | s/2 save | r/3 reset target | p/4 pause | t change tracker",
            (10, output.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA
        )
        if status == "TARGET LOST":
            cv2.putText(
                output, "Target lost. Press r or 3 to select a new target.",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
            )

    return output


def save_report_row(writer, frame_number, bbox, status, fps):
    x, y, w, h = ([int(v) for v in bbox] if bbox is not None else ["", "", "", ""])
    writer.writerow([frame_number, x, y, w, h, status, round(fps, 4)])


def initialize_tracking(frame, bbox, tracker_name):
    """Create and init tracker. Used on first selection and manual reset."""
    tracker = create_tracker(tracker_name)
    tracker.init(frame, bbox)
    return {
        "tracker":       tracker,
        "initial_bbox":  bbox,
        "current_bbox":  bbox,
        "last_valid_bbox": bbox,
        "status":        "TRACKING",
        "lost_counter":  0,
    }


# =========================
# Main Program
# =========================

print("====================================================")
print("User-Selected Object Tracking")
print("====================================================")
print(f"Video: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}\nMake sure it exists inside the data folder.")

input_fps = cap.get(cv2.CAP_PROP_FPS) or 25

ret, first_frame = cap.read()
if not ret or first_frame is None:
    raise RuntimeError("Could not read first frame.")

first_frame = resize_frame(first_frame, MAX_WIDTH)

print("\nSelect the target object tightly.")
print("Do not select a tiny dot. Select the full object.")
print("Press ENTER or SPACE after selecting.")

initial_bbox = select_target(first_frame, allow_cancel=False)
tracker_name = choose_tracker_type()
print(f"Selected tracker: {tracker_name}")

state = initialize_tracking(first_frame, initial_bbox, tracker_name)

output_video_path = os.path.join(OUTPUT_DIR, f"tracking_output_{tracker_name.lower()}_safe.mp4")
report_csv_path   = os.path.join(OUTPUT_DIR, f"tracking_report_{tracker_name.lower()}_safe.csv")

height, width = first_frame.shape[:2]
video_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    input_fps,
    (width, height)
)

csv_file   = open(report_csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame_number", "x", "y", "width", "height", "status", "fps"])

cv2.namedWindow("Tracking Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking Output", 1000, 650)

if SHOW_DEBUG_WINDOWS:
    for win in ("Preprocessing - Edge Detection", "Feature Extraction - Contours"):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 600, 400)

print("\nTracking started.")
print("q/1=quit | s/2=save | r/3=reset | p/4=pause | t=change tracker")

frame_number      = 1
saved_frame_count = 0
paused            = False
current_frame     = first_frame
output_frame      = first_frame

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video.")
            break

        frame_number += 1
        frame = resize_frame(frame, MAX_WIDTH)
        current_frame = frame

        # --- Preprocessing (only when debug windows are visible) ---
        if SHOW_DEBUG_WINDOWS:
            edges, contour_view = preprocess_frame(frame)
            cv2.imshow("Preprocessing - Edge Detection", edges)
            cv2.imshow("Feature Extraction - Contours", contour_view)

        # --- Tracking ---
        start_time = time.time()

        if state["status"] != "TARGET LOST":
            success, raw_bbox = state["tracker"].update(frame)

            # Decide if this update is valid
            tracking_ok = False
            if success:
                raw_bbox = tuple(int(v) for v in raw_bbox)
                fixed_bbox = stabilize_bbox_size(raw_bbox, state["last_valid_bbox"], state["initial_bbox"]) \
                    if STABILIZE_BOX_SIZE else raw_bbox

                if fixed_bbox and not is_bbox_lost(fixed_bbox, frame.shape[1], frame.shape[0]):
                    state["current_bbox"]    = fixed_bbox
                    state["last_valid_bbox"] = fixed_bbox
                    state["lost_counter"]    = 0
                    state["status"]          = "TRACKING"
                    tracking_ok = True

            if not tracking_ok:
                state["lost_counter"] += 1
                if state["lost_counter"] >= MAX_LOST_FRAMES:
                    state["status"]       = "TARGET LOST"
                    state["current_bbox"] = None
                else:
                    state["status"]       = "WARNING"
                    state["current_bbox"] = state["last_valid_bbox"]

        fps = 1.0 / (time.time() - start_time + 1e-6)

        output_frame = draw_output(
            frame, state["current_bbox"], state["status"],
            fps, frame_number, tracker_name
        )

        video_writer.write(output_frame)
        save_report_row(csv_writer, frame_number, state["current_bbox"], state["status"], fps)
        cv2.imshow("Tracking Output", output_frame)

    # --- Key handling ---
    key_code = cv2.waitKeyEx(30) & 0xFF

    if key_code in (ord("q"), ord("Q"), ord("1"), 27):
        print("Program stopped by user.")
        break

    elif key_code in (ord("s"), ord("S"), ord("2")):
        saved_frame_count += 1
        save_path = os.path.join(OUTPUT_DIR, f"saved_frame_{saved_frame_count}.jpg")
        cv2.imwrite(save_path, output_frame)
        print(f"Saved frame: {save_path}")

    elif key_code in (ord("p"), ord("P"), ord("4"), 32):
        paused = not paused
        print("Paused." if paused else "Resumed.")

    elif key_code in (ord("r"), ord("R"), ord("3")):
        paused = True
        new_bbox = select_target(current_frame, allow_cancel=True)
        if new_bbox:
            state = initialize_tracking(current_frame, new_bbox, tracker_name)
            print(f"Target reset using {tracker_name} tracker.")
        else:
            print("Reset cancelled.")
        paused = False

    elif key_code in (ord("t"), ord("T")):
        paused = True
        new_tracker = choose_tracker_type(current_tracker=tracker_name)
        if new_tracker != tracker_name:
            tracker_name = new_tracker
            if state["current_bbox"]:
                state = initialize_tracking(current_frame, state["current_bbox"], tracker_name)
                print(f"Tracker changed to {tracker_name} and reinitialized.")
            else:
                print(f"Tracker changed to {tracker_name}. Press r/3 to select a new target.")
        else:
            print(f"Tracker unchanged: {tracker_name}")
        paused = False


cap.release()
video_writer.release()
csv_file.close()
cv2.destroyAllWindows()

print("\nFinished.")
print(f"Output video: {output_video_path}")
print(f"Tracking report: {report_csv_path}")