import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="ssd_mobilenet_v1_coco_quant_postprocess.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load COCO class names
def load_class_names(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

class_names = load_class_names('coco.names')

# Preprocess the input image
def preprocess_image(image):
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(image, (input_shape[2], input_shape[1]))
    image_scaled = image_resized.astype(np.uint8)  # Ensure the image is in UINT8 format
    return np.expand_dims(image_scaled, axis=0)

# Run inference on the preprocessed image
def run_inference(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    return output_data

# Postprocess the output data to extract bounding boxes, class IDs, and scores
def postprocess_output(output_data):
    print(f"Output data shapes: {[data.shape for data in output_data]}")
    boxes = output_data[0][0]  # Extract the first batch
    class_ids = output_data[1][0]  # Extract the first batch
    scores = output_data[2][0]  # Extract the first batch
    return boxes, class_ids, scores

# Draw bounding boxes and labels on the image
def draw_boxes(image, boxes, class_ids, scores, threshold=0.5):
    height, width, _ = image.shape
    for i in range(len(scores)):
        if scores[i] > threshold:
            # Skip if class_id is NaN
            if np.isnan(class_ids[i]):
                print(f"Skipping detection with NaN class ID at index {i}.")
                continue

            class_id = int(class_ids[i])

            # Check if class_id is valid
            if not (0 <= class_id < len(class_names)):
                #print(f"Invalid class_id: {class_id}. Must be between 0 and {len(class_names) - 1}.")
                continue

            # Check for NaN values in the box coordinates
            if np.isnan(boxes[i]).any():
                print(f"Skipping box {boxes[i]} due to NaN values.")
                continue

            # Calculate bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]

            ymin = int(ymin * height)
            xmin = int(xmin * width)
            ymax = int(ymax * height)
            xmax = int(xmax * width)

            # Ensure that the coordinates are within the image dimensions
            ymin = max(0, min(height - 1, ymin))
            xmin = max(0, min(width - 1, xmin))
            ymax = max(0, min(height - 1, ymax))
            xmax = max(0, min(width - 1, xmax))

            # Draw the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = class_names[class_id]
            cv2.putText(image, f'{label} ({scores[i]:.2f})', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Main function to process live video feed
def main():
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        output_data = run_inference(frame)
        boxes, class_ids, scores = postprocess_output(output_data)

        # Draw results
        draw_boxes(frame, boxes, class_ids, scores)

        # Display the result
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
