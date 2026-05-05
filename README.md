# NeferQuality
This use case represents a radical shift from limited, error-prone human inspection to ad- vanced Artificial Intelligence (AI) systems that ensure product quality on production lines.
# Quality Control Classifier

## Overview

Automated quality inspection system using YOLO (best.pt) for product detection and EfficientNet (unified_classifier_advanced.pth) for defect classification.
unified_classifier_advanced drive link : https://drive.google.com/file/d/1g1XAu3NwY9WgwdzzqQWANj9uIlZK_tlH/view?usp=sharing
## Supported Products

- **Biscuit**: Defect_No, Defect_Color, Defect_Object, Defect_Shape
- **Bottle**: Good, Defective
- **Jar Lid**: Intact, Damaged
- **Package**: Intact, Damaged
- **Potato**: Good, Defective

## Output

- Green box = Good product
- Red box = Defective product
- Labels show product type and status

## How to Use

1. Run the application:
```bash
python Engine.py
```

2. Open browser

3. Click "Upload Image" and select a product image

4. Click "Classify" button

5. View the annotated result with detection box and status
