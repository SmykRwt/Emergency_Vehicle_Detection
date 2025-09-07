from roboflow import Roboflow
import cv2

# Initialize Roboflow model
rf = Roboflow(api_key="1rjU14yv8zWaiGEetWyY")  # Replace with your key
project = rf.workspace().project("object-recognition-d1oor-zifhx")  # Replace with your project name
model = project.version(1).model  # Replace `1` with your version

cap = cv2.VideoCapture(0)  # Webcam

def get_vehicle_count_and_draw_boxes(frame):
    cv2.imwrite("temp.jpg", frame)
    result = model.predict("temp.jpg", confidence=0.3).json()
    predictions = result['predictions']
    vehicle_count = len(predictions)

    for pred in predictions:
        x1 = int(pred['x'] - pred['width'] / 2)
        y1 = int(pred['y'] - pred['height'] / 2)
        x2 = int(pred['x'] + pred['width'] / 2)
        y2 = int(pred['y'] + pred['height'] / 2)

        # Draw box only
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return vehicle_count

green_light_duration = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count = get_vehicle_count_and_draw_boxes(frame)

    # Adjust green light duration based on vehicle count
    if count > 6:
        green_light_duration = 10
    elif count > 3:
        green_light_duration = 7
    else:
        green_light_duration = 5

    cv2.putText(frame, f"Vehicles: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Green light duration: {green_light_duration}s", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Traffic Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
