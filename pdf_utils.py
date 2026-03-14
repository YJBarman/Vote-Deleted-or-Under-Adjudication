import numpy as np
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
import gc

# ── Default grid constants ──
COLS      = 3
ROWS      = 10
DPI       = 300
HEADER_PX = 120
FOOTER_PX = 110
MARGIN_L  = 45
MARGIN_R  = 45


def get_total_pages(pdf_path: str) -> int:
    return pdfinfo_from_path(pdf_path)["Pages"]


def get_card_regions(img_np,
                     cols=COLS, rows=ROWS,
                     header_px=HEADER_PX, footer_px=FOOTER_PX,
                     margin_l=MARGIN_L, margin_r=MARGIN_R):
    H, W     = img_np.shape[:2]
    usable_w = W - margin_l - margin_r
    usable_h = H - header_px - footer_px
    cell_w   = usable_w // cols
    cell_h   = usable_h // rows
    regions  = []
    for r in range(rows):
        for c in range(cols):
            x = margin_l + c * cell_w
            y = header_px + r * cell_h
            w = cell_w if c < cols - 1 else (W - margin_r - x)
            h = cell_h if r < rows - 1 else (H - footer_px - y)
            regions.append((x, y, w, h))
    return regions


def iter_page_crops(pdf_path, first_page, last_page,
                    cols=COLS, rows=ROWS,
                    header_px=HEADER_PX, footer_px=FOOTER_PX,
                    margin_l=MARGIN_L, margin_r=MARGIN_R):
    """
    Generator — yields (page_num, card_idx, pil_crop) one at a time.
    Memory-efficient: loads and frees one page at a time.
    """
    for page_num in range(first_page, last_page + 1):
        pages   = convert_from_path(pdf_path, dpi=DPI,
                                     first_page=page_num,
                                     last_page=page_num)
        img_np  = np.array(pages[0])
        regions = get_card_regions(img_np, cols, rows,
                                   header_px, footer_px, margin_l, margin_r)
        for c_idx, (x, y, w, h) in enumerate(regions):
            crop = Image.fromarray(img_np[y:y+h, x:x+w].astype("uint8"))
            yield page_num, c_idx, crop

        del pages, img_np, regions
        gc.collect()
