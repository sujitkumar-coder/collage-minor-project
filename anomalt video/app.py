import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = r"good.mp4.mp4"   # <-- CHANGE VIDEO PATH HERE
PROJECT_TITLE = "Crowd Density Based Anomaly Detection"

GRID_SIZE = 5                 # 5x5 grid
DENSITY_THRESHOLD = 3.0       # std deviation threshold

# ==========================
# LOAD MODEL
# ==========================
model = YOLO("yolov8n.pt")

print(f"Starting {PROJECT_TITLE}")
print(f"Using video: {VIDEO_PATH}")

# ==========================
# OPEN VIDEO
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video {VIDEO_PATH}")
    raise SystemExit(1)

# ==========================
# ANNOTATORS
# ==========================
box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

# ==========================
# MAIN LOOP
# ==========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # YOLO detection + tracking
    results = model.track(frame, persist=True, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    # Filter only PERSON class (class_id = 0)
    if detections.class_id is not None:
        person_detections = detections[detections.class_id == 0]
    else:
        person_detections = detections[[]]

    total_people = len(person_detections)

    # ==========================
    # CROWD DENSITY GRID
    # ==========================
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
    grid_densities = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for box in person_detections.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cell_x = min(cx // cell_w, GRID_SIZE - 1)
        cell_y = min(cy // cell_h, GRID_SIZE - 1)

        grid_densities[cell_y, cell_x] += 1

    avg_density = float(np.mean(grid_densities))
    std_density = float(np.std(grid_densities))
    threshold = avg_density + DENSITY_THRESHOLD * std_density
    anomaly_count = int(np.sum(grid_densities > threshold))

    if anomaly_count > 0:
        print(
            f"ALERT: {anomaly_count} anomalous high-density zones detected! "
            f"Avg={avg_density:.6f}, Std={std_density:.6f}"
        )

    # ==========================
    # DRAW BOXES & LABELS
    # ==========================
    annotated = frame.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=person_detections)

    labels = [f"Person" for _ in range(total_people)]
    if total_people > 0:
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=person_detections,
            labels=labels
        )

    # ==========================
    # DRAW GRID HEATMAP
    # ==========================
    max_density = max(1.0, float(np.max(grid_densities)))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x, y = j * cell_w, i * cell_h
            density = grid_densities[i, j]

            intensity = min(255, int(255 * density / max_density))
            color = (0, intensity, 255 - intensity // 2)

            cv2.rectangle(
                annotated,
                (x, y),
                (x + cell_w, y + cell_h),
                color,
                2
            )

            if density > threshold:
                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + cell_w, y + cell_h),
                    (0, 0, 255),
                    3
                )

    # ==========================
    # TEXT OVERLAY
    # ==========================
    cv2.putText(annotated, f"Total People: {total_people}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(annotated, f"Avg Density: {avg_density:.2f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(annotated, f"Std Density: {std_density:.2f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(annotated, f"Anomalies: {anomaly_count}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if anomaly_count > 0 else (0, 255, 0), 2)

    cv2.putText(annotated, PROJECT_TITLE, (30, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ==========================
    # DISPLAY
    # ==========================
    cv2.imshow(PROJECT_TITLE, annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
print("Processing complete.")