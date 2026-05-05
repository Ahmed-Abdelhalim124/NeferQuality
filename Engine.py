# %%
print("Installing dependencies...")
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio", "-q"])
print("✓ Setup complete\n")

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

YOLO_MODEL_PATH = r'D:\Neferx\Quality\best.pt'
CLASSIFIER_MODEL_PATH = r'D:\Neferx\Quality\unified_classifier_advanced.pth'
CLASS_IMG_SIZE = 288
ALL_CLASSES = [
    'biscuit_Defect_Color',
    'biscuit_Defect_No',
    'biscuit_Defect_Object',
    'biscuit_Defect_Shape',
    'bottle_Defective',
    'bottle_Good',
    'jar_lid_Damaged',
    'jar_lid_Intact',
    'package_Damaged',
    'package_Intact',
    'potato_Defective',
    'potato_Good'
]
NUM_CLASSES = len(ALL_CLASSES)
CRITICAL_WEAK = ['jar_lid_Damaged', 'jar_lid_Intact']
MODERATE_WEAK = ['bottle_Defective', 'package_Damaged', 'package_Intact']
ALL_WEAK = CRITICAL_WEAK + MODERATE_WEAK
POSITIVE_STATUSES = ['Good', 'Intact', 'Defect_No']
NEGATIVE_STATUSES = ['Defective', 'Damaged', 'Defect_Color', 'Defect_Object', 'Defect_Shape']

class MultiHeadEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b2(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(0.3)
        self.main_classifier = nn.Linear(in_features, num_classes)
        self.aux_classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x, return_aux=False):
        features = self.backbone(x)
        features = self.dropout(features)
        main_out = self.main_classifier(features)
        if return_aux and self.training:
            aux_out = self.aux_classifier(features)
            return main_out, aux_out
        return main_out

def predict_with_tta(model, image, device, n_augments=5):
    model.eval()
    predictions = []
    
    if image.dim() != 4 or image.shape[1] != 3:
        print(f"Warning: Invalid input shape {image.shape} for TTA, using original image only")
        with torch.no_grad():
            pred = model(image.to(device))
            predictions.append(torch.softmax(pred, dim=1))
        return torch.stack(predictions).mean(dim=0)
    
    with torch.no_grad():
        pred = model(image.to(device))
        predictions.append(torch.softmax(pred, dim=1))
    
    tta_transforms = [
        A.Compose([A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Rotate(limit=10, p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    ]
    
    try:
        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Warning: Failed to convert tensor for TTA: {e}, using original image only")
        return torch.stack(predictions).mean(dim=0)
    
    for transform in tta_transforms[:n_augments-1]:
        try:
            aug_img = transform(image=img_np)['image']
            with torch.no_grad():
                pred = model(aug_img.unsqueeze(0).to(device))
                predictions.append(torch.softmax(pred, dim=1))
        except Exception as e:
            print(f"Warning: TTA augmentation failed: {e}, skipping")
            continue
    
    return torch.stack(predictions).mean(dim=0)

def get_classifier_transforms():
    return A.Compose([
        A.Resize(CLASS_IMG_SIZE, CLASS_IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def load_models():
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier_model = MultiHeadEfficientNet(NUM_CLASSES).to(device)
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        raise FileNotFoundError(f"Classifier model not found at {CLASSIFIER_MODEL_PATH}")
    classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    classifier_model.eval()
    
    return yolo_model, classifier_model, device

def predict_image(image):
    try:
        yolo_model, classifier_model, device = load_models()
        
        if image is None:
            return None, "Error: No image provided"
        
        image_np = np.array(image)
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        results = yolo_model(image_bgr, verbose=False)
        if len(results[0].boxes) == 0:
            return image_np, "Error: No product detected in the image"
        
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        h, w = image_bgr.shape[:2]
        x1 = max(0, x1 - 30)
        y1 = max(0, y1 - 30)
        x2 = min(w, x2 + 30)
        y2 = min(h, y2 + 30)
        
        if x2 <= x1 or y2 <= y1:
            return image_np, "Error: Invalid crop dimensions from YOLO detection"
        
        cropped = image_bgr[y1:y2, x1:x2]
        
        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return image_np, "Error: Cropped image is empty"
        
        transform = get_classifier_transforms()
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image=cropped_rgb)['image']
        image_tensor = image_tensor.unsqueeze(0)
        
        if image_tensor.shape != (1, 3, CLASS_IMG_SIZE, CLASS_IMG_SIZE):
            return image_np, f"Error: Invalid tensor shape {image_tensor.shape}"
        
        if any(cls in ALL_WEAK for cls in ALL_CLASSES):
            outputs = predict_with_tta(classifier_model, image_tensor, device, n_augments=5)
        else:
            with torch.no_grad():
                outputs = classifier_model(image_tensor.to(device))
        
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = ALL_CLASSES[predicted.item()]
        
        product_type, status = predicted_class.split('_', 1)
        product_type = product_type.capitalize()
        
        box_color = (0, 255, 0) if status in POSITIVE_STATUSES else (0, 0, 255)
        
        thickness = max(3, int(min(w, h) / 200))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, thickness)
        
        label_product = f"{product_type}"
        label_status = f"{status}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, min(w, h) / 800)
        text_thickness = max(2, int(font_scale * 2))
        
        (text_width_product, text_height_product), baseline_product = cv2.getTextSize(
            label_product, font, font_scale, text_thickness
        )
        (text_width_status, text_height_status), baseline_status = cv2.getTextSize(
            label_status, font, font_scale, text_thickness
        )
        
        padding = 10
        total_text_height = text_height_product + text_height_status + 3 * padding + baseline_product
        max_text_width = max(text_width_product, text_width_status)
        
        text_x = x1 + padding
        text_y_start = y1 + padding
        
        overlay = image_bgr.copy()
        cv2.rectangle(
            overlay,
            (x1 + thickness, y1 + thickness),
            (x1 + max_text_width + 3 * padding, y1 + total_text_height + padding),
            box_color,
            -1
        )
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, image_bgr)
        
        text_color = (255, 255, 255)
        cv2.putText(
            image_bgr,
            label_product,
            (text_x, text_y_start + text_height_product),
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA
        )
        cv2.putText(
            image_bgr,
            label_status,
            (text_x, text_y_start + text_height_product + text_height_status + padding),
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA
        )
        
        output_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        return output_image, None
    
    except Exception as e:
        return image_np if 'image_np' in locals() else None, f"Error: {str(e)}"

def create_gradio_interface():
    with gr.Blocks(title="Quality Control Classifier with YOLO") as interface:
        gr.Markdown("# Quality Control Classifier with YOLO Detection\nUpload an image to detect and classify the product type and its quality status. The output will show the image with a bounding box around the detected product, annotated with the product type and status.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                submit_button = gr.Button("Classify")
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Annotated Image")
                error_output = gr.Textbox(label="Error Message (if any)", visible=True)
        
        submit_button.click(
            fn=predict_image,
            inputs=image_input,
            outputs=[image_output, error_output]
        )
    
    return interface

if __name__ == "__main__":
    print("Launching Gradio interface...")
    interface = create_gradio_interface()
    interface.launch()



# %%
