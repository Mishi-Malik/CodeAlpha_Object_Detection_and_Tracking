import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ==============================
# YOLO MODEL
# ==============================
model = YOLO("yolo11n.pt")


# ==============================
# DEEP SORT TRACKER
# ==============================
tracker = DeepSort(
    max_age=15,
    n_init=3,
    nms_max_overlap=0.5,
    max_cosine_distance=0.3,
    nn_budget=100
)


# ==============================
# DISPLAY ID MAPPING
# ==============================
id_map = {}


# ==============================
# MENU
# ==============================
print("\n================================")
print(" Object Detection & Tracking")
print("================================")
print("1. Webcam")
print("2. Video File")
print("================================")

choice = input("Enter choice (1 or 2): ")


# ==============================
# INPUT
# ==============================
if choice == "1":

    cap = cv2.VideoCapture(0)

elif choice == "2":

    video_path = input("Enter video file path: ")
    cap = cv2.VideoCapture(video_path)

else:

    print("Invalid choice!")
    exit()


# ==============================
# CHECK INPUT
# ==============================
if not cap.isOpened():

    print("Error: Could not open webcam/video")
    exit()


# ==============================
# CREATE OUTPUT FOLDER
# ==============================
os.makedirs("output", exist_ok=True)


# ==============================
# VIDEO WRITER
# ==============================
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


output_path = "output/tracked_output.avi"

fourcc = cv2.VideoWriter_fourcc(*"MJPG")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)


# ==============================
# CHECK WRITER
# ==============================
if not writer.isOpened():

    print("Error: Could not create output video.")

    cap.release()
    exit()


print("\n================================")
print(" Tracking Started")
print("================================")
print("Press Q to quit.")
print("Output will be saved as:")
print(output_path)
print("================================")


# ==============================
# MAIN LOOP
# ==============================
while True:

    success, frame = cap.read()

    if not success:

        print("\nVideo finished.")
        break


    # ==============================
    # YOLO DETECTION
    # ==============================
    results = model(
        frame,
        verbose=False
    )


    detections = []


    # ==============================
    # GET DETECTIONS
    # ==============================
    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])


            # Confidence threshold
            if confidence < 0.60:
                continue


            x1, y1, x2, y2 = box.xyxy[0].tolist()


            class_id = int(box.cls[0])

            class_name = model.names[class_id]


            width_box = x2 - x1
            height_box = y2 - y1


            detections.append(
                (
                    [x1, y1, width_box, height_box],
                    confidence,
                    class_name
                )
            )


    # ==============================
    # DEEP SORT TRACKING
    # ==============================
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )


    # ==============================
    # DRAW TRACKS
    # ==============================
    for track in tracks:


        if not track.is_confirmed():
            continue


        deep_sort_id = track.track_id


        # Create simple IDs
        if deep_sort_id not in id_map:

            id_map[deep_sort_id] = len(id_map) + 1


        track_id = id_map[deep_sort_id]


        # Get bounding box
        l, t, r, b = track.to_ltrb()


        l = int(l)
        t = int(t)
        r = int(r)
        b = int(b)


        # Object name
        object_name = track.get_det_class()


        if object_name is None:
            object_name = "Object"


        # ==============================
        # DRAW BOUNDING BOX
        # ==============================
        cv2.rectangle(
            frame,
            (l, t),
            (r, b),
            (0, 255, 0),
            2
        )


        # ==============================
        # LABEL
        # ==============================
        label = f"{object_name} | ID: {track_id}"


        cv2.putText(
            frame,
            label,
            (l, max(t - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ==============================
    # DISPLAY
    # ==============================
    cv2.imshow(
        "Object Detection and Tracking",
        frame
    )


    # ==============================
    # SAVE PROCESSED FRAME
    # ==============================
    writer.write(frame)


    # ==============================
    # PRESS Q TO EXIT
    # ==============================
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==============================
# RELEASE EVERYTHING
# ==============================
writer.release()

cap.release()

cv2.destroyAllWindows()


# ==============================
# FINISHED
# ==============================
print("\n================================")
print(" Program finished successfully!")
print("================================")
print("Saved output video:")
print(output_path)
print("================================")