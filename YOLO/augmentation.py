import tensorflow as tf
import os
import glob

print("TensorFlow sürümü:", tf.__version__)

# Klasör Yolları
source_dir = 'C:\\Users\\hsisec\\Desktop\\foto'
target_dir = 'C:\\Users\\hsisec\\Desktop\\aug_foto'
os.makedirs(target_dir, exist_ok=True)

# Resimleri Bul
resim_uzantilari = ['*.jpg', '*.jpeg', '*.png']
resim_listesi = []
for uzanti in resim_uzantilari:
    resim_listesi.extend(glob.glob(os.path.join(source_dir, uzanti)))
    resim_listesi.extend(glob.glob(os.path.join(source_dir, uzanti.upper())))

print(f"Bulunan resim sayısı: {len(resim_listesi)}")

# ---------- BOYUT AYARI ----------
target_size = (600, 600)  # 🟢 600×600 yaptık!
# ---------------------------------

print(f"Hedef boyut: {target_size[0]}×{target_size[1]}")

# Efektler
rotate_angles = [1, 2, 3, 4]  # 90°, 180°, 270°, 360°
dark_factor = 0.3
bright_factor = 0.3
blur_sigma = 1.0

print("\n--- EFEKT PLANI ---")
print(f"1. ROTATE: {len(rotate_angles)} varyasyon (90°, 180°, 270°)")
print(f"2. DARK: 1 varyasyon (karartma)")
print(f"3. BRIGHT: 1 varyasyon (aydınlatma)")
print(f"4. BLUR: 1 varyasyon (bulanıklaştırma)")
print(f"Her orijinalden: {len(rotate_angles) + 3} varyasyon")
print(f"Toplam: {len(resim_listesi)} × {len(rotate_angles) + 3} = {len(resim_listesi) * (len(rotate_angles) + 3)} resim")
print("-" * 50)

def apply_rotate(image, k):
    """90 derece döndürme"""
    return tf.image.rot90(image, k=k)

def apply_dark(image, factor=0.3):
    """Karartma"""
    image = tf.expand_dims(image, 0)
    image = tf.keras.layers.RandomBrightness(factor=-factor)(image, training=True)
    return tf.squeeze(image, 0)

def apply_bright(image, factor=0.3):
    """Aydınlatma"""
    image = tf.expand_dims(image, 0)
    image = tf.keras.layers.RandomBrightness(factor=factor)(image, training=True)
    return tf.squeeze(image, 0)

def apply_blur(image, sigma=1.0):
    """Bulanıklaştırma"""
    image = tf.expand_dims(image, 0)
    try:
        image = tf.image.gaussian_filter2d(image, sigma=sigma)
    except:
        kernel = tf.ones((3, 3, 3, 1)) / 9.0
        image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(image, 0)

sayac = 0
for img_path in resim_listesi:
    print(f"İşleniyor: {os.path.basename(img_path)}")
    
    # Resmi yükle (ŞİMDİ 600×600 olacak)
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    
    # ----- 1. ROTATE -----
    for r in rotate_angles:
        rotated = apply_rotate(img_array, r)
        
        rotated = tf.clip_by_value(rotated, 0, 255)
        rotated = tf.cast(rotated, tf.uint8)
        pil_img = tf.keras.utils.array_to_img(rotated)
        save_path = os.path.join(target_dir, f"rotate_{r*90}deg_{os.path.basename(img_path)}")
        pil_img.save(save_path, quality=95)  # Kalite yüksek
        sayac += 1
    
    # ----- 2. DARK -----
    dark_img = apply_dark(img_array, factor=dark_factor)
    dark_img = tf.clip_by_value(dark_img, 0, 255)
    dark_img = tf.cast(dark_img, tf.uint8)
    pil_img = tf.keras.utils.array_to_img(dark_img)
    save_path = os.path.join(target_dir, f"dark_{os.path.basename(img_path)}")
    pil_img.save(save_path, quality=95)
    sayac += 1
    
    # ----- 3. BRIGHT -----
    bright_img = apply_bright(img_array, factor=bright_factor)
    bright_img = tf.clip_by_value(bright_img, 0, 255)
    bright_img = tf.cast(bright_img, tf.uint8)
    pil_img = tf.keras.utils.array_to_img(bright_img)
    save_path = os.path.join(target_dir, f"bright_{os.path.basename(img_path)}")
    pil_img.save(save_path, quality=95)
    sayac += 1
    
    # ----- 4. BLUR -----
    blur_img = apply_blur(img_array, sigma=blur_sigma)
    blur_img = tf.clip_by_value(blur_img, 0, 255)
    blur_img = tf.cast(blur_img, tf.uint8)
    pil_img = tf.keras.utils.array_to_img(blur_img)
    save_path = os.path.join(target_dir, f"blur_{os.path.basename(img_path)}")
    pil_img.save(save_path, quality=95)
    sayac += 1
    
    print(f"  ✓ {os.path.basename(img_path)} için {len(rotate_angles) + 3} varyasyon (600×600)")

print(f"\n✅ TOPLAM {sayac} resim oluşturuldu!")
print(f"📁 {target_dir}")
print(f"📏 Boyut: {target_size[0]}×{target_size[1]}")