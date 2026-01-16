import cv2
import numpy as np
from pathlib import Path
import os
from PIL import Image, ImageSequence

class KoreanTextPreprocessorV3:

    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.debug_dir = Path("/content/debug_output_v7")
        if debug_mode:
            self.debug_dir.mkdir(exist_ok=True, parents=True)

    # --- Debug ---
    def _save_debug_image(self, image, name):
        if self.debug_mode:
            try:
                debug_path = self.debug_dir / f"{name}.png"
                cv2.imwrite(str(debug_path), image)
            except Exception as e:
                print(f"Warning: Could not save debug image {name}: {str(e)}")

    # --- Pipeline xử lý ảnh ---
    def enhance_local_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge([cl, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        self._save_debug_image(enhanced, "1_clahe")
        return enhanced

    def denoise(self, image, method='gaussian'):
        if method == 'bilateral':
            denoised = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
        else:
            denoised = cv2.GaussianBlur(image, (3, 3), 0)
        self._save_debug_image(denoised, "2_denoised")
        return denoised

    def sharpen(self, image, kernel_type='light'):
        """
        kernel_type: 'light' (nhẹ), 'strong' (mạnh, mặc định cũ)
        """
        if kernel_type == 'light':
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
        else:
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        self._save_debug_image(sharpened, f"3_sharpened_{kernel_type}")
        return sharpened

    def adaptive_threshold(self, image, method='gaussian'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self._save_debug_image(gray, "4_gray")
        th = cv2.ADAPTIVE_THRESH_MEAN_C if method == 'mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        binary = cv2.adaptiveThreshold(gray, 255, th, cv2.THRESH_BINARY, blockSize=25, C=5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        self._save_debug_image(binary, "5_adaptive_threshold")
        return binary

    # --- Deskew ---
    def getSkewAngle(self, cvImage) -> float:
        gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=5)
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largestContour = max(contours, key=cv2.contourArea)
        minAreaRect = cv2.minAreaRect(largestContour)
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    def rotateImage(self, cvImage, angle: float):
        (h, w) = cvImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(cvImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def deskew(self, cvImage):
        angle = self.getSkewAngle(cvImage)
        if abs(angle) < 0.90:
            deskewed = self.rotateImage(cvImage, -1.0 * angle)
            self._save_debug_image(deskewed, "0_deskewed")
            return deskewed
        return cvImage

    # --- Smart pick frame ---
    def choose_best_frame(self, frames, max_frames=10):
        best_score = -1
        best_frame = frames[0]
        for i, f in enumerate(frames[:max_frames]):
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            score = cv2.countNonZero(binary)
            if score > best_score:
                best_score = score
                best_frame = f
        return best_frame

    # --- Xử lý ảnh / GIF / WebP / JPG động ---
    def process_image(self, image_path, output_path=None, denoise_method='gaussian',
                      threshold_method='gaussian', enable_deskew=True,
                      process_all_frames=False, smart_pick=True):
        image_path = Path(image_path)
        if not image_path.exists():
            raise ValueError(f"File does not exist: {image_path}")

        # luôn dùng Pillow để mở, để detect ảnh động kể cả .jpg
        img = Image.open(str(image_path))
        is_animated = getattr(img, "is_animated", False)

        frames = []
        if is_animated:
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                if not process_all_frames and i > 9:  # giới hạn 10 frame để tiết kiệm
                    break
                frame_cv = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
                if enable_deskew:
                    frame_cv = self.deskew(frame_cv)
                frames.append(frame_cv)

            if smart_pick:
                image_cv = self.choose_best_frame(frames)
            else:
                image_cv = frames[0]
        else:
            image_cv = cv2.imread(str(image_path))
            if enable_deskew and image_cv is not None:
                image_cv = self.deskew(image_cv)

        if image_cv is None:
            raise ValueError(f"Could not load image: {image_path}")

        # pipeline preprocess
        processed = self.enhance_local_contrast(image_cv)
        processed = self.denoise(processed, method=denoise_method)
        processed = self.sharpen(processed)
        processed = self.adaptive_threshold(processed, method=threshold_method)
        # processed = self.sharpen(processed)

        # save nếu cần
        if output_path:
            ext_out = Path(output_path).suffix.lower()
            if ext_out not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]:
                ext_out = ".png"
            output_path = str(output_path).rsplit('.', 1)[0] + ext_out
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(processed).save(output_path)

        return processed

    # --- Xử lý thư mục ---
    def process_directory(self, input_dir, output_dir, process_all_frames=False, smart_pick=True):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']
        all_files = []
        for ext in extensions:
            all_files.extend(input_dir.glob(f'*{ext}'))

        total_files = len(all_files)
        processed_files, failed_files = 0, 0

        print(f"\nFound {total_files} images to process in {input_dir}")

        for i, input_path in enumerate(all_files, 1):
            try:
                output_path = output_dir / f"processed_{input_path.name}"
                print(f"\nProcessing [{i}/{total_files}]: {input_path.name}")
                self.process_image(input_path, output_path,
                                   process_all_frames=process_all_frames,
                                   smart_pick=smart_pick)
                print(f"Success - Saved to: {output_path}")
                processed_files += 1
            except Exception as e:
                failed_files += 1
                print(f"Error processing {input_path.name}: {str(e)}")
                continue

        print(f"\nProcessing completed:")
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {processed_files}")
        print(f"Failed: {failed_files}")


# --- Main chạy thẳng ---
def main():
    preprocessor = KoreanTextPreprocessorV3(debug_mode=True)

    base_dir = Path("/content/drive/MyDrive/OFFICIAL_TEST_FOR_PHASE1/TEST_FOR_PHASE1")
    input_dir = base_dir /  "images_hyecho"/"TCA20172_00.jpg"
    output_dir = base_dir / "OutputImages" / "images_hyecho_demo"

    # Nếu muốn xử lý toàn bộ frame GIF, set process_all_frames=True
    preprocessor.process_directory(input_dir, output_dir,
                                   process_all_frames=False,
                                   smart_pick=True)


if __name__ == "__main__":
    main()