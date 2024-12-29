from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n.pt")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Predict on the current frame
    results = model.predict(source=frame)

    # Count the number of persons
    person_count = 0

    # Display the results
    for r in results:
        for box in r.boxes:
            if box.cls == 0:  # Assuming class 0 is 'person'
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                confidence = float(box.conf)  # Convert tensor to float
                cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the person count on the frame
    cv2.putText(frame, f'Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with predictions
    cv2.imshow('YOLO Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()