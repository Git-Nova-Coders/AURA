import cv2
import torch
import torchvision
import numpy as np

class PersonDetector:
    """
    Standalone Person Detector.
    Acts as a placeholder module to extract human crops from a full frame.
    In the future, the 'ANN Object Detection' project will replace this component.
    """
    def __init__(self, confidence_threshold=0.7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        # Load a lightweight pre-trained object detector from torchvision
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

    def detect_and_crop(self, frame: np.ndarray):
        """
        Receives a full BGR frame, detects people, and returns a list of cropped person images
        along with their bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []

        # Convert BGR to RGB and scale to 0-1 for torchvision
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.tensor(rgb_frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        tensor_frame = tensor_frame.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(tensor_frame)[0]

        person_crops = []
        
        # COCO dataset class index for 'person' is 1
        for i in range(len(predictions['labels'])):
            if predictions['labels'][i] == 1 and predictions['scores'][i] >= self.confidence_threshold:
                box = predictions['boxes'][i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                
                # Ensure box bounds are within the frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                # Crop the person
                crop = frame[y1:y2, x1:x2]
                
                # Only keep valid crops
                if crop.size > 0:
                    person_crops.append({
                        "crop": crop,
                        "box": (x1, y1, x2, y2),
                        "confidence": float(predictions['scores'][i].cpu().numpy())
                    })
                    
        return person_crops
