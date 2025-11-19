"""
=============================================================================
TABLE CELL CROPPER PIPELINE - FINAL FILTERED COLUMNS (1, 2, 3, 5, 7, 9)
=============================================================================
Features:
- Intersection-based cell detection
- Advanced image preprocessing (deskew, perspective correction)
- Filters strictly for columns 1, 2, 3, 5, 7, 9 by default.
- Saves all selected columns into a single unified folder.
- CLI interface
=============================================================================
"""

import cv2
import numpy as np
import os
import uuid
import time
import base64
import argparse
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Pipeline configuration settings"""
    PADDING = 6
    CELL_PADDING = 8  # Padding around cell crops
    HEADER_ROWS = 4   # Number of header rows to skip
    # UPDATED: Default columns to process: 1, 2, 3, 5, 7, 9
    DEFAULT_COLS = [1, 2, 3, 5, 7, 9]

# =============================================================================
# IMAGE ROTATION & ORIENTATION
# =============================================================================

def rotate_image(image, angle):
    """Placeholder for image rotation (if necessary)"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def detect_text_orientation(image_bgr):
    """Detects optimal table orientation using structure analysis."""
    if image_bgr is None:
        return 0

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape[:2]
    scale = 0.4
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 5)

    candidates = {}
    
    # Simplified orientation check focused on horizontal line presence
    for angle in [0, 90, 180, -90]:
        test_img = binary if angle == 0 else rotate_image(binary, angle)
        h, w = test_img.shape

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 50), 1))
        h_lines = cv2.morphologyEx(test_img, cv2.MORPH_OPEN, h_kernel)
        h_score = np.sum(h_lines > 0) / (test_img.size or 1)

        candidates[angle] = h_score

    best_angle = max(candidates, key=candidates.get)
    return best_angle


def prepare_crisp_image(image_bgr):
    """Prepares image with rotation and perspective correction."""
    if image_bgr is None:
        return image_bgr, 0.0, 0, None, None, image_bgr

    prep_start = time.time()
    img = image_bgr.copy()

    # Detect and apply rotation
    rotated_angle = detect_text_orientation(img)
    if rotated_angle != 0:
        img = rotate_image(img, rotated_angle)

    img_after_rotation = img.copy()

    # Convert to grayscale and exposure normalize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2:
        gray_exposed = np.clip((gray - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
    else:
        gray_exposed = gray

    # Deskew using Hough lines
    deskew_angle = 0.0
    try:
        edges = cv2.Canny(gray_exposed, 30, 100, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=gray.shape[1]//4, maxLineGap=10)
        if lines is not None and len(lines) > 5:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < 15:
                        angles.append(angle)
            if angles:
                deskew_angle = np.median(angles)
                if abs(deskew_angle) > 0.5:
                    h, w = gray_exposed.shape
                    center = (w // 2, h // 2)
                    Mrot = cv2.getRotationMatrix2D(center, deskew_angle, 1.0)
                    img_after_rotation = cv2.warpAffine(img_after_rotation, Mrot, (w, h),
                                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    gray_exposed = cv2.warpAffine(gray_exposed, Mrot, (w, h),
                                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"[SKEW] Skew detection failed: {e}")
        deskew_angle = 0.0

    # Find largest quadrilateral (main table)
    def find_largest_quad(img_gray):
        bl = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, thr = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_inv = 255 - thr
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(thr_inv, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = np.array([a[0] for a in approx], dtype="float32")
            s = pts.sum(axis=1)
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect
        return None

    quad = find_largest_quad(gray_exposed)
    M = None
    Minv = None
    warped_orig = img_after_rotation.copy()

    if quad is not None:
        (tl, tr, br, bl) = quad
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxW = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxH = int(max(heightA, heightB))
        if maxW > 50 and maxH > 50:
            dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(quad, dst)
            Minv = cv2.getPerspectiveTransform(dst, quad)
            warped_orig = cv2.warpPerspective(img_after_rotation, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
            gray_exposed = cv2.warpPerspective(gray_exposed, M, (maxW, maxH), flags=cv2.INTER_CUBIC)

    # Binarize for line detection
    binary = cv2.adaptiveThreshold(gray_exposed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize=31, C=15)
    if np.mean(binary) < 127:
        binary = 255 - binary

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    img_enhanced = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)

    print(f"[TIMING] Preprocessing: {time.time() - prep_start:.2f}s")
    return img_enhanced, deskew_angle, rotated_angle, M, Minv, warped_orig

# =============================================================================
# CELL IMAGE PREPROCESSING (CROPPING & CLEANING)
# =============================================================================

def preprocess_cell_image(cell_crop_bgr):
    """
    Basic binarization and sharpening for better viewing/external OCR.
    Applies an aggressive scale-up.
    """
    if cell_crop_bgr is None or cell_crop_bgr.size == 0:
        return cell_crop_bgr
    
    # Scale up for better visualization/future OCR (2.5x is common)
    scale_factor = 2.5
    cell_crop_bgr = cv2.resize(cell_crop_bgr, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    if len(cell_crop_bgr.shape) == 3:
        gray = cv2.cvtColor(cell_crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_crop_bgr.copy()
    
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure text is dark on white
    if np.mean(binary) < 127:
        binary = 255 - binary
    
    # Sharpen
    kernel_sharp = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(binary, -1, kernel_sharp)
    
    # Convert back to BGR
    final = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return final


# =============================================================================
# TABLE STRUCTURE DETECTION
# =============================================================================

def preprocess_for_table_detection(image_bgr):
    """Extract horizontal and vertical line masks."""
    t0 = time.time()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize=31, C=12)
    inv = 255 - binary

    horiz_len = max(30, W // 30)
    vert_len = max(30, H // 30)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    horiz_mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    horiz_mask = cv2.morphologyEx(horiz_mask, cv2.MORPH_CLOSE, small)
    vert_mask = cv2.morphologyEx(vert_mask, cv2.MORPH_CLOSE, small)

    combined = cv2.bitwise_or(horiz_mask, vert_mask)

    print(f"[TIMING] Table detection: {time.time() - t0:.2f}s")
    return horiz_mask, vert_mask, combined


def find_line_intersections(horiz_mask, vert_mask, img_shape):
    """
    Find grid intersection points by analyzing where horizontal and vertical lines cross.
    """
    H, W = img_shape[:2]
    
    # Create intersection mask
    intersection_mask = cv2.bitwise_and(horiz_mask, vert_mask)
    
    # Find contours in intersection mask
    contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    intersections = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            intersections.append((cx, cy))
    
    if not intersections:
        return [], [], []
    
    # Cluster intersections to find unique rows and columns
    intersections.sort(key=lambda p: (p[1], p[0]))  # Sort by Y then X
    
    # Extract unique Y coordinates (horizontal lines)
    y_coords = sorted(list(set([p[1] for p in intersections])))
    horiz_lines = []
    
    if len(y_coords) > 1:
        # Merge close Y values
        horiz_lines = [y_coords[0]]
        for y in y_coords[1:]:
            if y - horiz_lines[-1] > 15:  # Merge threshold
                horiz_lines.append(y)
            else:
                horiz_lines[-1] = (horiz_lines[-1] + y) // 2
    else:
        horiz_lines = y_coords
    
    # Extract unique X coordinates (vertical lines)
    x_coords = sorted(list(set([p[0] for p in intersections])))
    vert_lines = []
    
    if len(x_coords) > 1:
        # Merge close X values
        vert_lines = [x_coords[0]]
        for x in x_coords[1:]:
            if x - vert_lines[-1] > 15:  # Merge threshold
                vert_lines.append(x)
            else:
                vert_lines[-1] = (vert_lines[-1] + x) // 2
    else:
        vert_lines = x_coords
    
    # Add boundaries if missing
    if horiz_lines and horiz_lines[0] > 10:
        horiz_lines = [0] + horiz_lines
    if horiz_lines and horiz_lines[-1] < H - 10:
        horiz_lines.append(H)
    
    if vert_lines and vert_lines[0] > 10:
        vert_lines = [0] + vert_lines
    if vert_lines and vert_lines[-1] < W - 10:
        vert_lines.append(W)
    
    return intersections, horiz_lines, vert_lines


def detect_cells_from_intersections(horiz_lines, vert_lines, img_shape, header_rows=4):
    """Build cell grid from intersection-derived line coordinates."""
    H, W = img_shape[:2]
    
    if len(horiz_lines) < 2 or len(vert_lines) < 2:
        return [], [], 0
    
    all_boxes = []
    
    for r in range(len(horiz_lines) - 1):
        y1 = horiz_lines[r]
        y2 = horiz_lines[r + 1]
        
        raw_height = y2 - y1
        if raw_height < 15: # Skip tiny rows
            continue
        
        for c in range(len(vert_lines) - 1):
            x1 = vert_lines[c]
            x2 = vert_lines[c + 1]
            
            raw_width = x2 - x1
            if raw_width < 20: # Skip tiny columns
                continue
            
            # Calculate adaptive border to exclude table lines
            border_h = max(3, int(raw_height * 0.10))
            border_w = max(3, int(raw_width * 0.08))
            
            x_start = x1 + border_w
            y_start = y1 + border_h
            w = max(1, raw_width - 2 * border_w)
            h = max(1, raw_height - 2 * border_h)
            
            all_boxes.append({
                'x': x_start,
                'y': y_start,
                'w': w,
                'h': h,
                'row': r,
                'col': c,
                'raw_x1': x1,
                'raw_y1': y1,
                'raw_x2': x2,
                'raw_y2': y2
            })
    
    header_end_row = min(header_rows, len(horiz_lines) - 1)
    data_boxes = [b for b in all_boxes if b['row'] >= header_end_row]
    
    return data_boxes, all_boxes, header_end_row


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_image(filepath, result_dir, selected_columns=None):
    """Cell cropping pipeline without OCR, with filtered column saving."""
    start_time = time.time()
    stages = {}
    
    session_id = uuid.uuid4().hex
    short_session_id = session_id[:8]

    unique_result_dir = os.path.join(result_dir, short_session_id)
    os.makedirs(unique_result_dir, exist_ok=True)
    
    img = cv2.imread(filepath)
    if img is None:
        return {'stages': stages, 'results': [], 'error': 'Image loading failed.'}

    H_orig, W_orig = img.shape[:2]
    stages['image_shape'] = (int(W_orig), int(H_orig))

    # Preprocessing
    img_line_ready, deskew_angle, rotated_angle, M_persp, Minv_persp, warped_orig = prepare_crisp_image(img)
    cv2.imwrite(os.path.join(unique_result_dir, "0_corrected_image.png"), warped_orig)
    H, W = warped_orig.shape[:2]
    
    if abs(deskew_angle) > 0.5:
        stages['deskew_angle_deg'] = float(deskew_angle)
    if rotated_angle != 0:
        stages['major_rotation_applied'] = int(rotated_angle)
    if M_persp is not None:
        stages['perspective_applied'] = True

    # Table detection
    horiz_mask, vert_mask, combined_mask = preprocess_for_table_detection(img_line_ready)

    # Save debug masks
    cv2.imwrite(os.path.join(unique_result_dir, "horiz_mask.png"), horiz_mask)
    cv2.imwrite(os.path.join(unique_result_dir, "vert_mask.png"), vert_mask)
    cv2.imwrite(os.path.join(unique_result_dir, "combined_mask.png"), combined_mask)
    
    # Find grid intersections
    intersections, horiz_lines, vert_lines = find_line_intersections(horiz_mask, vert_mask, (H, W))
    
    # Detect cells - SKIP HEADER ROWS
    HEADER_ROWS = Config.HEADER_ROWS
    data_boxes, all_boxes, header_end_row = detect_cells_from_intersections(
        horiz_lines, vert_lines, (H, W), header_rows=HEADER_ROWS
    )

    # --- COLUMN FILTERING AND SETUP ---
    # Use the selected columns list, or the default, which now excludes 0
    ALL_SELECTED_COLS = set(selected_columns or Config.DEFAULT_COLS)

    # Filter data_boxes based on selected columns
    data_boxes = [box for box in data_boxes if box['col'] in ALL_SELECTED_COLS]
    stages['column_filter_applied'] = sorted(list(ALL_SELECTED_COLS))
    print(f"[FILTER] Processing columns: {stages['column_filter_applied']}")
    
    stages['detected_cells_count_raw'] = len(all_boxes)
    stages['detected_cells_count_data'] = len(data_boxes)
    stages['header_end_row'] = header_end_row

    # Create a single directory for saving all filtered crops
    data_crops_dir_name = "data_crops_1_2_3_5_7_9" # UPDATED FOLDER NAME
    output_dir = os.path.join(unique_result_dir, data_crops_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save debug visual with boxes
    dbg = warped_orig.copy()
    for box in all_boxes: # Draw all boxes first
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        color = (0, 165, 255) # Orange for header/unselected
        
        if box['row'] >= header_end_row and box['col'] in ALL_SELECTED_COLS:
            # All selected data columns are marked green
            color = (0, 255, 0)
                
        cv2.rectangle(dbg, (x, y), (x + w, y + h), color, 2)
        label = f"R{box['row']}C{box['col']}"
        cv2.putText(dbg, label, (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw header separator line (in red)
    if header_end_row < len(horiz_lines):
        header_y = horiz_lines[header_end_row]
        cv2.line(dbg, (0, header_y), (W, header_y), (0, 0, 255), 3)

    cv2.imwrite(os.path.join(unique_result_dir, "boxes_debug.png"), dbg)
    
    # Process cells
    print(f"Cropping and saving {len(data_boxes)} cells...")
    final_cell_results = []
    CELL_PADDING = Config.CELL_PADDING
    
    for box in data_boxes:
        r_idx = box['row']
        c_idx = box['col']
        x, y, w, h = box['x'], box['y'], box['w'], box['h']

        crop_type = 'filtered_data' 
        
        # Crop cell content WITH PADDING
        x0, y0 = max(0, x - CELL_PADDING), max(0, y - CELL_PADDING)
        x1, y1 = min(W, x + w + CELL_PADDING), min(H, y + h + CELL_PADDING)
        
        if x1 <= x0 or y1 <= y0:
            continue
            
        crop_orig = warped_orig[y0:y1, x0:x1].copy()

        if crop_orig is None or crop_orig.size == 0:
            continue

        # Preprocess crop (aggressive cleaning/scaling)
        crop_prep = preprocess_cell_image(crop_orig.copy())

        # Save crops
        base_name = f"R{r_idx}C{c_idx}"
        orig_name = f"{base_name}_orig.png"
        prep_name = f"{base_name}_prep.png"
        
        orig_path = os.path.join(output_dir, orig_name)
        prep_path = os.path.join(output_dir, prep_name)
        
        try:
            cv2.imwrite(orig_path, crop_orig)
            cv2.imwrite(prep_path, crop_prep)
        except Exception as e:
            print(f"⚠️ Failed to save crops R{r_idx}C{c_idx}: {e}")
            continue

        # Calculate display_row (0-indexed for data rows)
        display_row = r_idx - header_end_row

        final_cell_results.append({
            'coords': (x, y, w, h),
            # Use relative path component for JSON
            'raw_crop_file': os.path.join(data_crops_dir_name, orig_name),
            'preprocessed_crop_file': os.path.join(data_crops_dir_name, prep_name),
            'row_index': r_idx,
            'col_index': c_idx,
            'display_row': display_row,
            'crop_type': crop_type,
            'is_selected_for_main_data': True 
        })

    stages['cells_saved'] = len(final_cell_results)
    stages['total_processing_time'] = f"{time.time() - start_time:.2f}s"

    print(f"\n✅ Total processing time: {stages['total_processing_time']}")
    print(f"✅ Cropped and saved {len(final_cell_results)} cells.")

    return {
        'stages': stages,
        'results': final_cell_results,
        'session_id': short_session_id,
        'result_dir': os.path.join(result_dir, short_session_id),
        'source_file': os.path.basename(filepath),
    }

# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

def parse_args():
    """Parses command line arguments."""
    p = argparse.ArgumentParser(description="Table Cell Cropper Pipeline (Filtered Columns)")
    p.add_argument("--image", required=True, help="Path to input image file.")
    p.add_argument("--out", default="cropped_results", help="Output directory for results (will be created if it doesn't exist).")
    # UPDATED DEFAULT COLUMNS TO EXCLUDE 0
    default_cols_str = ",".join(map(str, Config.DEFAULT_COLS))
    p.add_argument("--columns", default=default_cols_str, help=f"Comma-separated list of 0-indexed column numbers to process (default: '{default_cols_str}').")
    p.add_argument("--header-rows", type=int, default=Config.HEADER_ROWS, help=f"Number of header rows to skip (default: {Config.HEADER_ROWS}).")
    return p.parse_args()

def main():
    """Main function to run the pipeline from CLI."""
    args = parse_args()
    
    # Update Config with CLI arguments
    Config.HEADER_ROWS = args.header_rows

    # Process selected columns
    selected_columns = None
    if args.columns:
        try:
            selected_columns = [int(x.strip()) for x in args.columns.split(',')]
        except ValueError:
            print("Error: --columns must be a comma-separated list of integers.")
            import sys; sys.exit(1)

    print(f"--- Starting Cell Cropper Pipeline for {args.image} ---")
    print(f"Output Directory: {args.out}")
    print(f"Selected Columns: {selected_columns}")
    print(f"Header Rows to Skip: {Config.HEADER_ROWS}")

    result = process_image(args.image, args.out, selected_columns)
    
    # Save a final summary/result JSON in the session folder
    if 'session_id' in result:
        session_dir = result['result_dir']
        final_json_path = os.path.join(session_dir, 'final_result.json')
        
        try:
            # Clean and summarize results for JSON output
            clean_results = [{
                'coords': r['coords'],
                'row': r['row_index'],
                'col': r['col_index'],
                'crop_type': r['crop_type'],
                'is_main_data': r['is_selected_for_main_data'],
                # Paths are now relative to the session directory
                'raw_crop': r['raw_crop_file'],
                'preprocessed_crop': r['preprocessed_crop_file'],
            } for r in result['results']]
            
            summary = {
                'session_id': result['session_id'],
                'result_dir': result['result_dir'],
                'source_file': result['source_file'],
                'stages': result['stages'],
                'cropped_cells_summary': clean_results
            }

            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Pipeline finished. Full results saved to: {final_json_path}")
        except Exception as e:
            print(f"⚠️ Could not save final JSON summary: {e}")
    else:
        print("\n❌ Pipeline failed before generating a session ID.")

if __name__ == "__main__":
    main()