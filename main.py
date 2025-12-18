import pytesseract
import cv2
import re
from collections import defaultdict
from pathlib import Path
from rapidfuzz import process
from horses import VALID_HORSES

# ---- Tesseract path (Windows) ----
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---- Regex: match scores even if commas/spaces missing ----
SCORE_REGEX = re.compile(r"(\d{1,3}(?:[,\s]\d{3})+|\d{1,5})")

# ---- Helpers ----
def normalize(text: str) -> str:
    return text.replace("\n", " ").replace("  ", " ").strip()

# ---- OCR Parsing ----
def parse_score_image(image_path):
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    gray = gray[:, :int(w * 0.75)]
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    text = pytesseract.image_to_string(gray)
    text = normalize(text)

    results = []
    found_horses = set()

    # Split text into chunks
    chunks = re.split(r"[,\n]", text)

    for chunk in chunks:
        chunk = normalize(chunk)
        if not chunk:
            continue

        score_match = SCORE_REGEX.search(chunk)
        if not score_match:
            continue

        score_str = score_match.group(1).replace(",", "").replace(" ", "")
        try:
            score = int(score_str)
        except ValueError:
            continue

        horse_candidate = chunk.replace(score_match.group(0), "").strip()
        if not horse_candidate:
            continue

        best = process.extractOne(horse_candidate, VALID_HORSES)
        if not best:
            continue

        horse, confidence, _ = best
        min_conf = 50

        if horse in found_horses:
            continue

        if confidence >= min_conf:
            results.append((horse, score))
            found_horses.add(horse)

    return results

# ---- Aggregation ----
def compute_averages(image_folder):
    totals = defaultdict(int)
    counts = defaultdict(int)

    for image_path in Path(image_folder).glob("*.jpg"):
        parsed = parse_score_image(image_path)
        found = set()

        for horse, score in parsed:
            if horse not in found:
                totals[horse] += score
                counts[horse] += 1
                found.add(horse)

    averages = {horse: totals[horse] / counts[horse] for horse in totals}
    return averages, counts

# ---- Entry Point ----
if __name__ == "__main__":
    folder = "screenshots"  # change to your folder path
    averages, counts = compute_averages(folder)

    print("\nAverage Score Per Horse:\n")
    for horse in sorted(averages, key=averages.get, reverse=True):
        print(f"{horse}: {averages[horse]:.2f}  (photos: {counts[horse]})")
