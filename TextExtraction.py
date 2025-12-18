import easyocr
from PIL import Image
import os

reader = easyocr.Reader(['ar', 'en'], gpu=False)   
def Extract(image_path):
    if not os.path.exists(image_path):
        print("there is not photo")
        return
    
    print(f"Extract text -> {image_path}")
    result = reader.readtext(image_path)
    text = " ".join([line[1] for line in result])
    print("\nSuccessfully!\n" + "═"*80)
    print(text)
    print("═"*80)
 
    
    txt_file = image_path.rsplit(".", 1)[0] + ".txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved → {txt_file}")


Extract("WhatsApp Image 2025-12-06 at 19.13.26_b2dd7518.jpg")