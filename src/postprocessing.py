
from konlpy.tag import Okt
from pathlib import Path


def segment_korean_text(text: str, tokenizer=None) -> str:
    """
    text: chuỗi OCR tiếng Hàn
    tokenizer: instance Okt hoặc Mecab
    return: chuỗi đã tách từ bằng khoảng trắng
    """
    if tokenizer is None:
        tokenizer = Okt()
    words = tokenizer.morphs(text)  # tách từ
    return " ".join(words)


def improve_korean_ocr(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(" File đầu vào không tồn tại:", input_file)
        return


    lines = input_file.read_text(encoding="utf-8").splitlines()


    okt = Okt()

    improved_lines = []
    for line in lines:
        line_clean = " ".join(line.split())  # chuẩn hóa khoảng trắng
        improved_line = segment_korean_text(line_clean, okt)
        improved_lines.append(improved_line)


    output_file.write_text("\n".join(improved_lines), encoding="utf-8")
    print(f" File đã được lưu tại: {output_file}")


input_drive_path = "/content/drive/MyDrive/OFFICIAL_TEST_FOR_PHASE1/TEST_FOR_PHASE1/OutputOCR_ver4/images_hyecho_demo_processed/processed_A000000173589_03_res.txt"
output_drive_path = "/content/drive/MyDrive/input_ocr_A000000173589_03_segmented.txt"
improve_korean_ocr(input_drive_path, output_drive_path)
