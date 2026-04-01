
import cv2
import torch
import numpy as np
import supervision as sv
from collections import defaultdict, deque

def track_waste_video(model_path, input_video, output_video,
                      conf_thresh=0.3, confirm_frames=5):

    # Load YOLOv5 model
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=model_path,
        trust_repo=True
    )
    model.conf = conf_thresh

    tracker = sv.ByteTrack()

    WASTE_CLASSES = [
        'Battery', 'Glass', 'Medical', 'Metal',
        'Organic', 'Paper', 'Plastic', 'SmartPhone'
    ]

    cap = cv2.VideoCapture(input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # ✅ FIX: handle bad FPS
    if fps == 0:
        fps = 30

    print("Width:", width, "Height:", height, "FPS:", fps)

    # ✅ FIX: safe codec
    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    print("VideoWriter opened:", out.isOpened())

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    confirmed_counts = {cls: set() for cls in WASTE_CLASSES}
    history = defaultdict(lambda: deque(maxlen=confirm_frames))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        results = model(frame)
        pred = results.xyxy[0].cpu().numpy()

        if len(pred) > 0:
            detections = sv.Detections(
                xyxy=pred[:, :4],
                confidence=pred[:, 4],
                class_id=pred[:, 5].astype(int)
            )
        else:
            detections = sv.Detections.empty()

        # Tracking
        detections = tracker.update_with_detections(detections)

        # Labels
        labels = []
        for cls, conf, tid in zip(
            detections.class_id,
            detections.confidence,
            detections.tracker_id
        ):
            labels.append(f"{model.names[int(cls)]} {conf:.2f} ID:{tid}")

        # Draw boxes
        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, labels)

        # Counting logic
        for cls, tid in zip(detections.class_id, detections.tracker_id):
            if tid is None:
                continue

            label = model.names[int(cls)]

            if label in WASTE_CLASSES:
                history[tid].append(label)

                if len(history[tid]) == confirm_frames:
                    most_common = max(
                        set(history[tid]),
                        key=history[tid].count
                    )

                    if tid not in confirmed_counts[most_common]:
                        confirmed_counts[most_common].add(tid)

        # Display counts
        y = 30
        for cls in WASTE_CLASSES:
            count = len(confirmed_counts[cls])
            cv2.putText(frame, f"{cls}: {count}",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            y += 25

        total = sum(len(confirmed_counts[c]) for c in WASTE_CLASSES)
        cv2.putText(frame, f"Total: {total}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    print("✅ Tracking complete!")
    print("Saved at:", output_video)
