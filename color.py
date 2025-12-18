from rembg import remove
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

input_path = r"c:\Users\User\Music\download (5).jpeg" 


original = Image.open(input_path).convert("RGBA")
no_bg = remove(original, alpha_matting=True)


img = no_bg.resize((300, 300))
data = np.array(img)
mask = data[:, :, 3] > 50
pixels = data[mask][:, :3]

filtered_pixels = [p for p in pixels if not (np.all(p > 230) or np.all(p < 40))]
if len(filtered_pixels) < 500: 
    filtered_pixels = pixels


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(filtered_pixels)
labels = kmeans.labels_
dominant_idx = Counter(labels).most_common(1)[0][0]
r, g, b = kmeans.cluster_centers_[dominant_idx].astype(int)


COLORS = {
    (255,0,0): "Red", (139,0,0): "Dark Red", (178,34,34): "Firebrick",
    (255,165,0): "Orange", (255,215,0): "Gold",
    (255,255,0): "Yellow",
    (0,255,0): "Green", (50,205,50): "Lime Green", (34,139,34): "Forest Green", (128,128,0): "Olive",
    (0,255,255): "Cyan", (0,128,128): "Teal",
    (0,0,255): "Blue", (30,144,255): "Dodger Blue", (0,0,139): "Navy Blue", (25,25,112): "Midnight Blue",
    (128,0,128): "Purple", (148,0,211): "Violet",
    (255,192,203): "Pink", (255,20,147): "Hot Pink",
    (165,42,42): "Brown", (160,82,45): "Sienna",
    (0,0,0): "Black", (255,255,255): "White",
    (128,128,128): "Gray", (211,211,211): "Light Gray", (169,169,169): "Silver"
}

def get_color(rgb):
    r,g,b = rgb
    closest = min(COLORS, key=lambda c: (r-c[0])**2 + (g-c[1])**2 + (b-c[2])**2)
    name = COLORS[closest]
    
    brightness = (r*299 + g*587 + b*114)//1000
    if brightness < 70 and name != "Black":
        return f"Dark {name}"
    elif brightness > 200 and name != "White":
        return f"Light {name}"
    else:
        return name

final_color = get_color((r,g,b))

print(f"Dominant Color → {final_color}")
print(f"RGB            → ({r}, {g}, {b})")