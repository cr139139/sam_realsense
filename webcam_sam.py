from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import numpy as np

cam = cv2.VideoCapture(4)
model_type = "vit_t"
sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

if not cam.isOpened():
    print("error opening camera")
    exit()
import time
while True:
    start = time.time()
    # Capture frame-by-frame
    ret, frame = cam.read()
    # if frame is read correctly ret is True
    if not ret:
        print("error in retrieving frame")
        break

    h, w, _ = frame.shape

    predictor.set_image(frame)
    masks, scores, _ = predictor.predict(point_coords=np.array([[w // 2, h // 2]]), point_labels=np.array([1]))
    n_masks = len(masks)

    masked_image1 = frame * masks[0][:, :, np.newaxis]
    masked_image2 = frame * masks[1][:, :, np.newaxis]
    masked_image3 = frame * masks[2][:, :, np.newaxis]

    frame = cv2.circle(frame, (w // 2, h // 2), radius=10, color=(0, 0, 255), thickness=-1)
    image1 = np.concatenate([frame, masked_image1], axis=1)
    image2 = np.concatenate([masked_image2, masked_image3], axis=1)
    image = np.concatenate([image1, image2], axis=0)
    cv2.imshow('window', image)

    if cv2.waitKey(1) == ord('q'):
        break
    print(1/(time.time() - start))

cam.release()
cv2.destroyAllWindows()