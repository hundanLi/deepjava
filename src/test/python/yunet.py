import cv2
import time

image_path = "largest_selfie.jpg"

img = cv2.imread(image_path)

# Get image dimensions
img_W = int(img.shape[1])
img_H = int(img.shape[0])

detector = cv2.FaceDetectorYN.create("face_detection_yunet_2022mar.onnx", "", (320, 320))

# Save time
t0 = time.time()

# Getting the detections
detector.setInputSize((img_W, img_H))
detections = detector.detect(img)

# Calculate inference time
inf_time = round(time.time() - t0, 3)

# Print results
print(f"Detections: {detections}")
print(f"Inference time: {inf_time}s")


if (detections[1] is not None) and (len(detections[1]) > 0):
    for detection in detections[1]:
        # Converting predicted and ground truth bounding boxes to required format
        pred_bbox = detection
        pred_bbox = [int(i) for i in pred_bbox[:4]]

        cv2.rectangle(img,pred_bbox,(0,255,0),5)

# Write inference time
img = cv2.putText(img, f"Inf Time: {inf_time}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)


cv2.imwrite("detected-face.jpg", img)


# wget -O test-image.jpg "https://www.dropbox.com/s/cmf4bgwl559l4s2/face-detection-test-image.jpg?dl=1"