"""
Generates a synthetic histopathology-like sample image for testing the app.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

def generate_histopathology_image(save_path="sample_test_image.png", size=224):
    np.random.seed(42)

    # Pink/purple base background (like H&E stained tissue)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[:, :] = [235, 200, 220]  # light pink background

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Draw cell-like circular blobs (nuclei)
    num_cells = 60
    for _ in range(num_cells):
        x = np.random.randint(5, size - 5)
        y = np.random.randint(5, size - 5)
        r = np.random.randint(4, 12)

        # Dark purple nucleus
        color = (
            np.random.randint(80, 130),
            np.random.randint(20, 70),
            np.random.randint(100, 160)
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

        # Lighter cytoplasm ring
        r2 = r + np.random.randint(2, 5)
        cyto_color = (
            np.random.randint(180, 230),
            np.random.randint(130, 180),
            np.random.randint(180, 220)
        )
        draw.ellipse([x - r2, y - r2, x + r2, y + r2], outline=cyto_color, width=2)

    # Add some dense clusters (simulating IDC regions)
    for _ in range(5):
        cx = np.random.randint(40, size - 40)
        cy = np.random.randint(40, size - 40)
        for _ in range(8):
            ox = cx + np.random.randint(-15, 15)
            oy = cy + np.random.randint(-15, 15)
            r = np.random.randint(6, 14)
            draw.ellipse([ox-r, oy-r, ox+r, oy+r],
                         fill=(60, 10, 90),
                         outline=(100, 30, 120), width=1)

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Add subtle noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 6, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    img.save(save_path)
    print(f"Sample image saved: {save_path}")
    print(f"Now upload this image in the Streamlit app!")
    return save_path

if __name__ == "__main__":
    generate_histopathology_image()
