import google.generativeai as genai
from PIL import Image
import os


API_KEY = "AIzaSyDtjZ_05_xEqQto4P8vVzC9LZ82kNnpI5U" 
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash')


reader = None

def get_reader():
    global reader
    if reader is None:
        reader = model
    return reader

def extract_gemini(image_path):
    if not os.path.exists(image_path):
        print(f"خطأ: الصورة '{image_path}' مش موجودة! حطها في نفس مجلد الكود.")
        return None
    
    print(f"جاري استخراج النص من → {image_path}")
    try:
        img = Image.open(image_path)
        reader = get_reader()
        
        response = reader.generate_content([
            "استخرج كل النص الموجود في الصورة بدقة عالية جدًا، احتفظ باللغة الأصلية (عربي أو إنجليزي أو أي لغة) ورتب النص زي ما هو في الصورة. كن دقيقاً ولا تضيف أي نص إضافي:",
            img
        ])
        
        text = response.text.strip()
        
        print("\nتم بنجاح ✓\n" + "═"*80)
        print(text)
        print("═"*80)
        
        # حفظ النص تلقائي في ملف .txt
        txt_file = image_path.rsplit(".", 1)[0] + ".txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"تم الحفظ → {txt_file}")
        
        return text
    except Exception as e:
        print(f"خطأ في الاستخراج: {e}")
        print("تأكد من الـ API Key أو اتصال الإنترنت. جرب موديل آخر زي 'gemini-2.5-pro' لو ده مش شغال.")
        return None

# استخدمه على صورة واحدة
extract_gemini("download (5).jpeg")  # غير الاسم لو الصورة مختلفة

# مثال: استخراج من مجلد كامل (للـ batch سريع)
# def extract_from_folder(folder_path):
#     for file in os.listdir(folder_path):
#         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             extract_gemini(os.path.join(folder_path, file))
# 
# extract_from_folder("مجلد_الصور")