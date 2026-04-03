import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8m.pt")

# User input for video or camera
input_path = input("Enter video path (or 'camera' for webcam): ").strip()
if input_path.lower() == "camera":
    video_path = 0
    print("Using laptop camera...")
else:
    video_path = input_path or "good.mp4.mp4"
    print(f"Using video: {video_path}")

PROJECT_TITLE = "Crowd Density Based Anomaly Detection"
print(f"Starting {PROJECT_TITLE}")
print(f"Video: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")
    raise SystemExit(1)

box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

GRID_SIZE = 5
DENSITY_THRESHOLD = 3.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Track detections so IDs stay stable across frames
    results = model.track(frame, persist=True, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    # Keep only people
    if detections.class_id is not None:
        person_detections = detections[detections.class_id == 0]
    else:
        person_detections = detections[[]]

    total_people = len(person_detections)

    # Crowd density grid
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
            f"Avg: {avg_density:.8f}, Std: {std_density:.10f}"
        )

    # Draw boxes and labels
    annotated = frame.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=person_detections)

    labels = []
    for i in range(total_people):
        labels.append(f"person {i+1}")
    if len(labels) > 0:
        annotated = label_annotator.annotate(scene=annotated, detections=person_detections, labels=labels)

    # Draw density heatmap and anomalies
    max_density = max(1.0, float(np.max(grid_densities)))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x, y = j * cell_w, i * cell_h
            density = grid_densities[i, j]

            color_intensity = min(255, int(255 * density / max_density))
            color = (0, color_intensity, 255 - color_intensity // 2)

            cv2.rectangle(annotated, (x, y), (x + cell_w, y + cell_h), color, 2)

            if density > threshold:
                cv2.rectangle(annotated, (x, y), (x + cell_w, y + cell_h), (0, 0, 255), 3)

    # Text overlay
    cv2.putText(annotated, f"Total People: {total_people}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Density Avg: {avg_density:.2f} Std: {std_density:.2f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated, f"Anomalies: {anomaly_count}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if anomaly_count > 0 else (0, 255, 0), 2)
    cv2.putText(annotated, PROJECT_TITLE, (30, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow(PROJECT_TITLE, annotated)

    # ESC to stop
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")