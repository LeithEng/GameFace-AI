import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import json
import os

# --- 1. SETUP & ASSET LOADING ---
st.set_page_config(page_title="GameFace-AI", page_icon="⚔️", layout="wide")

@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO('yolov8m.pt') 
    
    # Load ResNet
    classifier = models.resnet18()
    classifier.fc = nn.Linear(classifier.fc.in_features, 9)
    # Update this path if necessary
    model_path = '03- Object detection and multi-universe classifier/models/witcher_sekiro_v8.pth'
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    with open('03- Object detection and multi-universe classifier/class_mapping.json', 'r') as f:
        mapping = json.load(f)
        
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
        
    return yolo, classifier, mapping, device, font

yolo, classifier, class_mapping, device, font = load_assets()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("🛠️ Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, max_value=1.0, value=0.80, step=0.05,
    help="Higher = More strict identification. Lower = More 'guesses'."
)

st.sidebar.divider()
st.sidebar.markdown("""
### 🎨 Color Guide
* 🟢 **Green**: Confirmed Hero
* 🟡 **Yellow**: Uncertain (Below Threshold)
* ⚪ **Grey**: NPC / Other
""")

# --- 3. MAIN UI ---
st.title("⚔️ GameFace-AI: Multiverse Classifier")
st.markdown("Bridge the gap between **The Witcher 3** and **Sekiro**.")

# File uploader
uploaded_file = st.file_uploader("Upload a screenshot", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    if st.button("⚡ Run Multi-Stage Detection"):
        with st.spinner("Analyzing Multiverse..."):
            results = yolo(input_image)
            draw = ImageDraw.Draw(input_image)
            
            img_width, img_height = input_image.size
            found_count = 0
            stats = {}

            # Processing loop
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0:  # YOLO 'person' class
                        found_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # ResNet Inference
                        crop = input_image.crop((x1, y1, x2, y2))
                        tensor_crop = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])(crop).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = classifier(tensor_crop)
                            probs = torch.nn.functional.softmax(output[0], dim=0)
                            score, pred = torch.max(probs, 0)
                        
                        # --- Logic & Labeling ---
                        raw_label = class_mapping.get(str(pred.item()), "Unknown")
                        
                        if score.item() < conf_threshold:
                            display_label = "Unknown/NPC"
                            box_color = "#FFCC00"  # Yellow
                        elif raw_label.lower() == "other":
                            display_label = "Other/NPC"
                            box_color = "#808080"  # Grey
                        else:
                            display_label = raw_label
                            box_color = "#00FF00"  # Green
                        
                        full_text = f"{display_label} ({score:.1%})"
                        stats[display_label] = stats.get(display_label, 0) + 1

                        # --- SMART POSITIONING ---
                        # If the box starts within 40px of the top, put label INSIDE
                        text_bbox = draw.textbbox((x1, y1), full_text, font=font)
                        text_w = text_bbox[2] - text_bbox[0]
                        text_h = text_bbox[3] - text_bbox[1]
                        
                        if y1 < 40:
                            label_y = y1 + 5  # Push inside the box
                        else:
                            label_y = y1 - text_h - 10 # Standard position above
                        
                        # Draw label background and text
                        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 5], fill=box_color)
                        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
                        draw.text((x1 + 5, label_y + 2), full_text, fill="black", font=font)

            # --- Display Results ---
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(input_image, use_container_width=True)
            
            with col2:
                st.subheader("📊 Analysis")
                if found_count > 0:
                    st.metric("Total People Found", found_count)
                    for label, count in stats.items():
                        st.write(f"**{label}:** {count}")
                else:
                    st.write("No one detected.")