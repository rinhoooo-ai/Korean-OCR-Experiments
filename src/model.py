from paddleocr import PaddleOCR
# ===== Config =====
INPUT_DIR = "/content/drive/MyDrive/OFFICIAL_TEST_FOR_PHASE1/TEST_FOR_PHASE1/OutputImages_ver4/images_hyecho_demo"    #  folder chứa ảnh
OUTPUT_DIR = "/content/drive/MyDrive/OFFICIAL_TEST_FOR_PHASE1/TEST_FOR_PHASE1/OutputOCR_ver4/images_hyecho_demo_processed"  #  folder lưu kết quả

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Initialize PaddleOCR =====
ocr = PaddleOCR(
    lang='korean',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# ===== Run OCR on all images in folder =====
img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",".gif",".webp"]

for img_file in Path(INPUT_DIR).glob("*"):
    if img_file.suffix.lower() in img_exts:
        print(f"Processing: {img_file.name}")
        result = ocr.predict(str(img_file))

        # Tạo folder con cho mỗi ảnh
        save_base = Path(OUTPUT_DIR) / img_file.stem
        os.makedirs(save_base, exist_ok=True)

        # Lưu kết quả
        for res in result:
            res.print()
            try:
                # Một số kết quả có font_size = 0 gây lỗi -> bắt và bỏ qua
                res.save_to_img(str(save_base))
            except ValueError as e:
                if "font size must be greater than 0" in str(e):
                    print(f"Bỏ qua save_to_img cho {img_file.name} (font size = 0)")
                else:
                    raise
            res.save_to_json(str(save_base))

print("Done! Check results in:", OUTPUT_DIR)