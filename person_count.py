import cv2
import numpy as np
def person_count():

        net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")

        # Define output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

        # Initialize video capture object
        cap = cv2.VideoCapture(0)

        # Loop over frames from webcam
        while True:
            # Capture frame
            ret, frame = cap.read()

            # Get image dimensions
            height, width, channels = frame.shape

            # Preprocess input image
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            # Set input to the network
            net.setInput(blob)

            # Forward pass through the network
            outs = net.forward(output_layers)

            # Initialize variables
            class_ids = []
            confidences = []
            boxes = []

            # Loop over each detection
            for out in outs:
                for detection in out:
                    # Get class ID and confidence score
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Get bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Append results to lists
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # Apply non-maximum suppression to remove redundant detections
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Count number of persons
            count = 0
            for i in indices:

                if class_ids[i] == 0:
                    count += 1

                    # Draw bounding box and label on frame
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display result
            cv2.imshow("Frame", frame)
            if(count>1):
                print("Cheating Detected")
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        # Release video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
person_count()