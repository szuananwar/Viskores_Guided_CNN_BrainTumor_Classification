import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

os.makedirs("docs/figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -----------------------------
# Accuracy Comparison
# -----------------------------
baseline_acc = 0.8844
fusion_acc = 0.9241

plt.figure(figsize=(7, 5))
plt.bar(
    ["Baseline CNN", "Viskores-Guided\nFusion CNN"],
    [baseline_acc * 100, fusion_acc * 100]
)

plt.ylabel("Accuracy (%)")
plt.title("Classification Accuracy Comparison")
plt.ylim(80, 100)

for i, v in enumerate([baseline_acc * 100, fusion_acc * 100]):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=12)

plt.tight_layout()
plt.savefig("results/final_accuracy_comparison.png", dpi=300)
plt.savefig("docs/figures/final_accuracy_comparison.png", dpi=300)
plt.close()

# -----------------------------
# Workflow Figure
# -----------------------------
W, H = 1800, 1100
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

def font(size, bold=False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

title_font = font(50, True)
box_font = font(26, True)
small_font = font(23, False)
arrow_font = font(42, True)

draw.rectangle([0, 0, W, 95], fill=(20, 70, 130))
draw.text((W//2, 50), "Viskores-Guided Feature Enhancement for CNN-Based Brain Tumor Classification",
          fill="white", anchor="mm", font=title_font)

def box(x, y, w, h, text, fill):
    draw.rounded_rectangle([x, y, x+w, y+h], radius=22, fill=fill, outline=(40, 40, 40), width=3)
    draw.text((x+w/2, y+h/2), text, fill="black", anchor="mm", font=box_font, align="center")

def arrow(x1, y1, x2, y2):
    draw.line([x1, y1, x2, y2], fill="black", width=4)
    draw.text(((x1+x2)/2, (y1+y2)/2 - 10), "➜", fill="black", anchor="mm", font=arrow_font)

# Pipeline 1
draw.text((450, 145), "Pipeline 1: Baseline CNN", fill=(20, 70, 130), anchor="mm", font=title_font)

box(180, 220, 540, 85, "2D MRI Images", (220, 235, 255))
box(180, 360, 540, 85, "CNN Classifier", (220, 255, 230))
box(180, 500, 540, 85, "4-Class Classification", (255, 240, 220))
box(180, 640, 540, 85, "Accuracy: 88.4%", (255, 230, 230))

arrow(450, 305, 450, 360)
arrow(450, 445, 450, 500)
arrow(450, 585, 450, 640)

# Pipeline 2
draw.text((1350, 145), "Pipeline 2: Viskores-Guided Fusion CNN", fill=(20, 70, 130), anchor="mm", font=title_font)

box(1050, 200, 600, 75, "2D MRI Images", (220, 235, 255))
box(1050, 300, 600, 75, "Simulated 3D Volume", (235, 225, 255))
box(1050, 400, 600, 75, "VTK Conversion", (235, 225, 255))
box(1050, 500, 600, 75, "Viskores Isosurface Analysis", (225, 245, 255))
box(1050, 600, 600, 75, "18 Geometric + Intensity Features", (255, 245, 220))
box(1050, 700, 600, 75, "Feature Fusion with CNN", (220, 255, 230))
box(1050, 800, 600, 75, "4-Class Classification", (255, 240, 220))
box(1050, 900, 600, 75, "Accuracy: 92.4%", (220, 255, 220))

for y in [275, 375, 475, 575, 675, 775, 875]:
    arrow(1350, y, 1350, y+25)

# Footer
draw.rounded_rectangle([180, 1000, 1650, 1070], radius=18, fill=(245, 250, 255), outline=(20, 70, 130), width=3)
draw.text(
    (915, 1035),
    "Result: Viskores-guided feature fusion improved CNN accuracy by approximately 4.0 percentage points.",
    fill="black",
    anchor="mm",
    font=small_font
)

img.save("docs/figures/workflow_overview.png")
img.save("results/workflow_overview.png")

# -----------------------------
# CNN Fusion Architecture Figure
# -----------------------------
W, H = 1600, 900
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

draw.rectangle([0, 0, W, 90], fill=(20, 70, 130))
draw.text((W//2, 45), "Viskores-Guided Feature Fusion CNN Architecture",
          fill="white", anchor="mm", font=title_font)

box(100, 180, 420, 90, "MRI Image Input\n128 × 128 × 3", (220, 235, 255))
box(100, 340, 420, 90, "CNN Branch\nConv + Pool + Dense", (220, 255, 230))

box(1050, 180, 420, 90, "Viskores Feature Input\n18 Features", (255, 245, 220))
box(1050, 340, 420, 90, "Feature Branch\nDense Layers", (255, 235, 210))

box(590, 520, 420, 90, "Concatenation\nImage + Viskores Features", (235, 225, 255))
box(590, 680, 420, 90, "Softmax Output\nGlioma / Meningioma / No Tumor / Pituitary", (255, 240, 220))

arrow(310, 270, 310, 340)
arrow(1260, 270, 1260, 340)
arrow(520, 385, 590, 550)
arrow(1050, 385, 1010, 550)
arrow(800, 610, 800, 680)

img.save("docs/figures/fusion_cnn_architecture.png")
img.save("results/fusion_cnn_architecture.png")

print("Saved figures:")
print("docs/figures/final_accuracy_comparison.png")
print("docs/figures/workflow_overview.png")
print("docs/figures/fusion_cnn_architecture.png")
