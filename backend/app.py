from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google.cloud import vision
from google.oauth2 import service_account
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from PIL import Image, ImageOps
import cv2
import io
import json
import traceback
import sys
import logging
import os
import uuid
import shutil
from table import process_image as process_table_image

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('flask_app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Replace with your Vision API credentials
SERVICE_ACCOUNT_JSON = {
      "type": "service_account",
  "project_id": "outstanding-yew-476610-h5",
  "private_key_id": "b9cfbac878b4fe15676cc59ea583ae42c1c5d269",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC1reIVcFLLUz4p\nI3ZTUWzpufMZS7fm57ZPV5PJgHg8ZebvqaZSCcflboU3N8tLHHvfjXN5gc+C443j\nVe9JVDeIbfCrcLoSAhbeY01T6hXjad2pEixfCM1V0Cl86/TU1HxrgfozZwIIliV2\nvXO8Qjj3D0k7VG1t+H8/hp5cHg2DXisCN5xgvjhvdc58GJIpZo3VsoJNXbUy3BM/\ng4KEp8SJLBT0IuH9ZfcE4qYZsNa/kHmpdJofI4CSefI3qTri8ePRA1hNMofKvhho\nTf8fbj6OfekCH8TjWX8iCxsY54OTlA/vfJKVipFeKo+YzNBZHR/a8Nxn53xVOXR/\n+KSg181RAgMBAAECggEAAmhip+CzL3CBlDlHr7fNOtVeHwEnY9qEPlPZrrkh4djw\nW8mR6b1sWVM5t0SN3J/y/Y/HtKQWOmcDfh34alMQJEyo5FQAuQEWzHORdQBEvlOm\nfnZu+xuLKRw+usyyB8sBBtZ7yZpe2tacSX7U7/9sNBWhg/5rVF+PjU4YZnfMiqq7\nLXCjb09iwu8sILm1Xkyi7BNdFHT/yTcmfWOE+xGQo3tmRx9Yj4A8so/HVSsY960M\n7ozCIuL7Ys09MZCHU4ex/lG536IswBkb4RxgNLam183rRMZ3+E4eidL00lAmnDY/\nxXnyjPSm0IZ0OO0dRBE/Siy80v18tzyRtK2ByRNtzwKBgQD1hfBiBuCDVdXSClHI\nzGAxt67+yU3DYKKsXBER5ISR2RzMxIMGqYHN7RB9J8wtRGjw+sGJmz2gC/+KL/Tn\no/1D/tBq54gL7/FcrwVWDjV/RkFz1n91xhyCIRExbOITlp2HX91q7od0N4nro+6W\nMom1NGeRQ5nLD48CUJtHavSCAwKBgQC9boWGtISVBqPyZafv4aUFjdr5vYrUq18q\nwYjbpB/Ggm7kaI0JkhJnscaq/8mwHAJ5//NutrMX87UccZ0dVuTK2X0jovHRd8PK\n1p61s3GKBDaVWHoPboiQ+bbkgTM8nBGJHLR1tg4CjPjivMY528ix/N4RblVCklId\n4qd/C/FdGwKBgCBkCTCFg70P5+OL5Po+rDoi531JhW7PIubmRoI7yZmMMRZ6nmaU\nmkFWkyRPycn3CnkwO7QxvNGOg6nxZbfhlJoR1eEkpngcsZTuqh+ORFSEKkJj+/DH\nsB8iyafhm3nGFwYzCXz/9vLIGPPzbph2FmDHhxpM9s8pQE6n37RuUc+NAoGBAJ1I\nugdptTfDhrbJ3xGWyhz9dpar4SzJicAHZ2nvMQ7y175AbPJIXY3Jlwn+TekyqH/B\nm88OVU+K45LVZr2om1kuEfBX5+6jQWcWojp93sTY0LVZ/Cb6ANxW6pt6Bx/I+epd\nchzWZ3WItVIVqYT6zv2x1nSkppgaw6HawnYDZYmHAoGAXdgTxw4T1yS2Zlk/uJeQ\nqcLpVRcSRCWUZYf/eop3ydbPnlf8qVAX0GfnfmHOB92dX8NeOX9TC992MMyUGTMc\nmqQbDyl1jYTS/hNHZUMyPo5SUvp3rSfxA79RKn/464dDnpDtvYJ2VQ1/w/AJLiaY\nnbzQ3iiw9RXGMn+poDig+Sw=\n-----END PRIVATE KEY-----\n",
  "client_email": "ocr-758@outstanding-yew-476610-h5.iam.gserviceaccount.com",
  "client_id": "116253747489591616579",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/ocr-758%40outstanding-yew-476610-h5.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

try:
    credentials = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_JSON)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    logger.info("Vision API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Vision API client: {e}")
    vision_client = None

def fix_image_orientation(image_bytes):
    """Fix image orientation based on EXIF data."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        original_format = image.format
        logger.debug(f"Image format: {original_format}, size: {image.size}")
        
        try:
            image = ImageOps.exif_transpose(image)
            logger.debug("Applied EXIF transpose")
        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"EXIF transpose failed: {e}, trying manual rotation")
            try:
                exif = image.getexif()
                if exif is not None:
                    orientation = exif.get(274)
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                    logger.debug(f"Applied manual rotation for orientation: {orientation}")
            except Exception as inner_e:
                logger.debug(f"Manual rotation failed: {inner_e}")
        
        output = io.BytesIO()
        if original_format and original_format.upper() in ['PNG', 'JPEG', 'JPG']:
            image.save(output, format=original_format)
        else:
            image.save(output, format='JPEG', quality=95)
        
        output.seek(0)
        result = output.getvalue()
        logger.debug(f"Fixed image size: {len(result)} bytes")
        return result
        
    except Exception as e:
        logger.error(f"Image orientation fix failed: {e}")
        logger.error(traceback.format_exc())
        return image_bytes


def enhance_handwritten_image(image_bytes):
    """Improve contrast and readability for handwritten Telugu text."""
    try:
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            return image_bytes

        height, width = image.shape[:2]
        scale_factor = 2.0 if max(height, width) < 2000 else 1.5
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
        inverted = cv2.bitwise_not(binary)
        final_image = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        success, buffer = cv2.imencode('.png', final_image)
        if not success:
            return image_bytes
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Handwriting enhancement failed: {e}")
        return image_bytes

def get_text_from_word(word):
    """Extract text from a word object"""
    try:
        return ''.join([symbol.text for symbol in word.symbols])
    except Exception as e:
        logger.error(f"Error extracting text from word: {e}")
        return ""

def is_handwritten_text(word):
    """Detect if text is handwritten based on characteristics"""
    try:
        if not word.symbols:
            return False
        
        text = get_text_from_word(word)
        if not text or not text.strip():
            return False
        
        text_lower = text.lower().strip()
        
        header_keywords = [
            'క్రమ', 'uid', 'సభ్యురాల్', 'పొదుపు', 'అప్పు', 'కట్టిన', 'మొత్తం', 'నిల్వ',
            'loan', 'bank', 'vo', 'cif', 'type', 'amount', 'serial', 'member', 'details',
            'స్త్రీనిధి', 'ఉన్నతి', 'కొత్త', 'మంజూర్', 'వసూళ్లు', 'చెల్లింపులు',
            'shg', 'బ్యాంక్', 'సమావేశం', 'జరిగిన', 'గత', 'నెల', 'నిల్వలు', 'అంతర్గత',
            'రికవరీ', 'వివరములు', 'క్యాపిటల్', 'ఖాతా', 'తేదీ', 'నాటికి'
        ]
        
        for kw in header_keywords:
            if kw in text_lower and len(text_lower) <= len(kw) + 5:
                return False
        
        has_numbers = bool(re.search(r'\d', text))
        is_mostly_numbers = bool(re.match(r'^[\d\s.,/-]+$', text))
        
        if is_mostly_numbers or (has_numbers and len(text.strip()) > 0):
            return True
        
        has_telugu = bool(re.search(r'[\u0C00-\u0C7F]', text))
        if has_telugu:
            return True
        
        return len(text.strip()) > 0
        
    except Exception as e:
        logger.debug(f"Error in is_handwritten_text: {e}")
        return True

def create_combined_html(main_df, ref_df=None, metadata=None):
    """Create HTML matching official SHG format"""
    html = []
    
    # Get meeting details from metadata
    shg_name = metadata.get('village', '.....................') if metadata else '.....................'
    meeting_date = metadata.get('meeting_date', '............') if metadata else '............'
    
    html.append('''
    <style>
        body {
            font-family: 'Noto Sans Telugu', Arial, sans-serif;
            margin: 20px;
            background: white;
        }
        .container {
            max-width: 100%;
            margin: 0 auto;
        }
        .header-section {
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #000;
            padding-bottom: 10px;
        }
        .header-title {
            font-size: 16px;
            font-weight: bold;
            margin: 5px 0;
        }
        .signature-boxes {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .signature-box {
            border: 2px solid #000;
            padding: 5px 15px;
            font-weight: bold;
            min-width: 200px;
            text-align: center;
        }
        .main-table, .reference-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 11px;
        }
        .main-table th, .main-table td,
        .reference-table th, .reference-table td {
            border: 1px solid #000;
            padding: 6px 4px;
            text-align: center;
            vertical-align: middle;
        }
        .main-table th {
            background-color: #e0e0e0;
            font-weight: bold;
            font-size: 10px;
        }
        .reference-table th {
            background-color: #e0e0e0;
            font-weight: bold;
        }
        .header-row-1 {
            background-color: #d0d0d0 !important;
            font-weight: bold;
        }
        .header-row-2 {
            background-color: #e0e0e0 !important;
            font-weight: bold;
            font-size: 10px;
        }
        .header-row-3 {
            background-color: #f0f0f0 !important;
            font-size: 9px;
        }
        .serial-col {
            width: 30px;
        }
        .uid-col {
            width: 80px;
        }
        .name-col {
            width: 120px;
        }
        .amount-col {
            width: 60px;
        }
        .total-row {
            background-color: #f5f5f5 !important;
            font-weight: bold;
        }
        .ref-serial {
            width: 40px;
            font-weight: bold;
        }
        .ref-desc {
            text-align: left !important;
            padding-left: 8px;
        }
        .signature-cell {
            height: 80px;
            vertical-align: middle !important;
        }
    </style>
    ''')
    
    html.append('<div class="container">')
    
    # Header with signature boxes
    html.append('<div class="signature-boxes">')
    html.append('<div class="signature-box">SHG స్టాంప్ & లీడర్స్ సంతకం</div>')
    html.append('<div class="signature-box">VOA సంతకం</div>')
    html.append('</div>')
    
    # Meeting header
    html.append('<div class="header-section">')
    html.append(f'<div class="header-title">……………………..{shg_name} స్వయం సహాయక సంఘం ………………………… తేదీ {meeting_date} న జరిగిన సమావేశ ఆర్థిక లావాదేవీల వివరాలు</div>')
    html.append('</div>')
    
    # MAIN TABLE
    if main_df is not None and len(main_df) > 0:
        html.append('<table class="main-table">')
        html.append('<thead>')
        
        # Row 1: Major headers
        html.append('<tr class="header-row-1">')
        html.append('<th rowspan="3" class="serial-col">క్రమ.సం.</th>')
        html.append('<th rowspan="3" class="uid-col">UID</th>')
        html.append('<th rowspan="3" class="name-col">సభ్యురాలు పేరు</th>')
        html.append('<th colspan="2">పొదుపు</th>')
        html.append('<th colspan="10">అప్పు రికవరీ వివరములు</th>')
        html.append('<th colspan="2">కొత్త అప్పు మంజూరు</th>')
        html.append('</tr>')
        
        # Row 2: Sub-headers
        html.append('<tr class="header-row-2">')
        html.append('<th rowspan="2">ఈ నెల పొదుపు</th>')
        html.append('<th rowspan="2">ఈ నెల వరకు పొదుపు నిల్వ</th>')
        html.append('<th colspan="2">SHG అంతర్గత అప్పు</th>')
        html.append('<th colspan="2">బ్యాంక్ అప్పు</th>')
        html.append('<th colspan="2">స్త్రీనిధి/HD/సీడ్ క్యాపిటల్</th>')
        html.append('<th colspan="2">ఉన్నతి</th>')
        html.append('<th colspan="2">CIF/VO అంతర్గత</th>')
        html.append('<th rowspan="2">అప్పు రకం</th>')
        html.append('<th rowspan="2">మొత్తం</th>')
        html.append('</tr>')
        
        # Row 3: Detailed sub-headers
        html.append('<tr class="header-row-3">')
        for i in range(5):
            html.append('<th class="amount-col">కట్టిన మొత్తం</th>')
            html.append('<th class="amount-col">అప్పు నిల్వ</th>')
        html.append('</tr>')
        
        html.append('</thead>')
        html.append('<tbody>')
        
        # Data rows (15 rows)
        for idx in range(len(main_df)):
            row_class = 'total-row' if idx == len(main_df) - 1 else ''
            html.append(f'<tr class="{row_class}">')
            for col in main_df.columns:
                val = main_df.iloc[idx][col]
                display = str(val).strip() if pd.notna(val) and str(val) not in ['', 'nan', 'None'] else ''
                html.append(f'<td>{display}</td>')
            html.append('</tr>')
        
        # Total row
        html.append('<tr class="total-row">')
        html.append('<td colspan="3">మొత్తం:</td>')
        for i in range(14):
            html.append('<td></td>')
        html.append('</tr>')
        
        html.append('</tbody>')
        html.append('</table>')
    
    # REFERENCE TABLE
    if ref_df is not None and len(ref_df) > 0:
        html.append('<table class="reference-table">')
        html.append('<thead>')
        html.append('<tr class="header-row-1">')
        html.append('<th colspan="3">ఈ సమావేశం లో జరిగిన వసూళ్లు మొత్తం</th>')
        html.append('<th colspan="2">ఈ సమావేశం లో జరిగిన చెల్లింపులు మొత్తం</th>')
        html.append('<th colspan="2">గత నెల బ్యాంక్ నిల్వలు</th>')
        html.append('</tr>')
        html.append('</thead>')
        html.append('<tbody>')
        
        for idx in range(len(ref_df)):
            row = ref_df.iloc[idx]
            is_total = row['క్రమ.సం.'] == 'మొత్తం:'
            row_class = 'total-row' if is_total else ''
            
            # Rows 9-11: Merged signature section
            if idx >= 8 and idx <= 10:
                if idx == 8:
                    html.append(f'<tr class="{row_class}">')
                    html.append(f'<td class="ref-serial">{row["క్రమ.సం."]}</td>')
                    html.append(f'<td class="ref-desc">{row["వసూళ్లు వివరం"]}</td>')
                    html.append(f'<td>{row["వసూళ్లు మొత్తం"]}</td>')
                    html.append(f'<td class="ref-desc">{row["చెల్లింపులు వివరం"]}</td>')
                    html.append(f'<td rowspan="3" colspan="3" class="signature-cell">')
                    html.append('<div style="text-align: center;">')
                    html.append('<div style="margin-bottom: 30px;">SHG స్టాంప్ & లీడర్స్ సంతకం</div>')
                    html.append('<div style="border-top: 1px solid #000; padding-top: 10px;">VOA సంతకం</div>')
                    html.append('</div>')
                    html.append('</td>')
                    html.append('</tr>')
                else:
                    html.append(f'<tr class="{row_class}">')
                    html.append(f'<td class="ref-serial">{row["క్రమ.సం."]}</td>')
                    html.append(f'<td class="ref-desc">{row["వసూళ్లు వివరం"]}</td>')
                    html.append(f'<td>{row["వసూళ్లు మొత్తం"]}</td>')
                    html.append(f'<td class="ref-desc">{row["చెల్లింపులు వివరం"]}</td>')
                    html.append('</tr>')
            else:
                html.append(f'<tr class="{row_class}">')
                html.append(f'<td class="ref-serial">{row["క్రమ.సం."]}</td>')
                html.append(f'<td class="ref-desc">{row["వసూళ్లు వివరం"]}</td>')
                html.append(f'<td>{row["వసూళ్లు మొత్తం"]}</td>')
                html.append(f'<td class="ref-desc">{row["చెల్లింపులు వివరం"]}</td>')
                html.append(f'<td>{row.get("చెల్లింపులు మొత్తం", "")}</td>')
                html.append(f'<td class="ref-desc">{row.get("బ్యాంక్ నిల్వలు వివరం", "")}</td>')
                html.append(f'<td>{row.get("బ్యాంక్ నిల్వలు విలువ", "")}</td>')
                html.append('</tr>')
        
        html.append('</tbody>')
        html.append('</table>')
    
    html.append('</div>')
    
    return ''.join(html)

def transform_to_official_format(df, metadata=None):
    """
    Transform extracted data into official SHG table format with:
    - Main table: 15 member rows × 17 columns
    - Reference table: 12 summary rows
    """
    
    # Official column headers for main table (17 columns)
    official_headers = [
        "క్రమ.సం.",  # Serial No
        "UID",
        "సభ్యురాలు పేరు",  # Member Name
        "ఈ నెల పొదుపు",  # This month savings
        "ఈ నెల వరకు పొదుపు నిల్వ",  # Total savings till date
        "SHG అంతర్గత అప్పు - కట్టిన మొత్తం",  # SHG Internal - Paid
        "SHG అంతర్గత అప్పు - అప్పు నిల్వ",  # SHG Internal - Balance
        "బ్యాంక్ అప్పు - కట్టిన మొత్తం",  # Bank Loan - Paid
        "బ్యాంక్ అప్పు - అప్పు నిల్వ",  # Bank Loan - Balance
        "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ - కట్టిన మొత్తం",  # Streenidhi - Paid
        "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ - అప్పు నిల్వ",  # Streenidhi - Balance
        "ఉన్నతి - కట్టిన మొత్తం",  # Unnathi - Paid
        "ఉన్నతి - అప్పు నిల్వ",  # Unnathi - Balance
        "CIF/VO అంతర్గత - కట్టిన మొత్తం",  # CIF/VO - Paid
        "CIF/VO అంతర్గత - అప్పు నిల్వ",  # CIF/VO - Balance
        "కొత్త అప్పు రకం",  # New loan type
        "కొత్త అప్పు మొత్తం"  # New loan amount
    ]
    
    # Map extracted columns to official positions
    column_mapping = {
        "UID": "UID",
        "సభ్యురాలు పేరు": "సభ్యురాలు పేరు",
        "ఈ నెల పొదుపు": "ఈ నెల పొదుపు",
        "SHG అంతర్గత అప్పు - కట్టిన మొత్తం": "SHG అంతర్గత అప్పు - కట్టిన మొత్తం",
        "బ్యాంక్ అప్పు - కట్టిన మొత్తం": "బ్యాంక్ అప్పు - కట్టిన మొత్తం",
        "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ - కట్టిన మొత్తం": "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ - కట్టిన మొత్తం"
    }
    
    # Create main table with 15 rows
    main_table_data = []
    
    for i in range(15):
        row = {"క్రమ.సం.": str(i + 1)}
        
        if i < len(df):
            # Fill with extracted data
            source_row = df.iloc[i]
            
            for source_col, target_col in column_mapping.items():
                if source_col in source_row.index and pd.notna(source_row[source_col]) and str(source_row[source_col]).strip():
                    row[target_col] = str(source_row[source_col]).strip()
                else:
                    row[target_col] = ""
        
        # Fill all remaining columns with empty strings
        for header in official_headers:
            if header not in row:
                row[header] = ""
        
        main_table_data.append(row)
    
    # Create DataFrame with official column order
    main_df = pd.DataFrame(main_table_data, columns=official_headers)
    
    # Calculate totals for reference table
    total_savings_this_month = 0
    total_shg_internal_paid = 0
    total_bank_paid = 0
    total_streenidhi_paid = 0
    
    for i in range(len(df)):
        # Calculate savings total
        if "ఈ నెల పొదుపు" in df.columns:
            savings_str = df.iloc[i].get("ఈ నెల పొదుపు", "0")
            try:
                savings_val = float(re.sub(r'[^\d.]', '', str(savings_str)))
                total_savings_this_month += savings_val
            except:
                pass
        
        # Calculate SHG internal total
        if "SHG అంతర్గత అప్పు - కట్టిన మొత్తం" in df.columns:
            shg_str = df.iloc[i].get("SHG అంతర్గత అప్పు - కట్టిన మొత్తం", "0")
            try:
                shg_val = float(re.sub(r'[^\d.]', '', str(shg_str)))
                total_shg_internal_paid += shg_val
            except:
                pass
        
        # Calculate bank loan total
        if "బ్యాంక్ అప్పు - కట్టిన మొత్తం" in df.columns:
            bank_str = df.iloc[i].get("బ్యాంక్ అప్పు - కట్టిన మొత్తం", "0")
            try:
                bank_val = float(re.sub(r'[^\d.]', '', str(bank_str)))
                total_bank_paid += bank_val
            except:
                pass
        
        # Calculate streenidhi total
        if "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ - కట్టిన మొత్తం" in df.columns:
            streenidhi_str = df.iloc[i].get("స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ - కట్టిన మొత్తం", "0")
            try:
                streenidhi_val = float(re.sub(r'[^\d.]', '', str(streenidhi_str)))
                total_streenidhi_paid += streenidhi_val
            except:
                pass
    
    # Total collections
    total_collections = total_savings_this_month + total_shg_internal_paid + total_bank_paid + total_streenidhi_paid
    
    # Reference table data (12 rows including total)
    reference_data = [
        {
            "క్రమ.సం.": "1",
            "వసూళ్లు వివరం": "పొదుపు వసూళ్లు మొత్తం",
            "వసూళ్లు మొత్తం": f"{total_savings_this_month:.2f}" if total_savings_this_month > 0 else "",
            "చెల్లింపులు వివరం": "VOకు చెల్లించిన పొదుపులు",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "ఖాతా వివరము",
            "బ్యాంక్ నిల్వలు విలువ": "తేదీ నాటికి రూ."
        },
        {
            "క్రమ.సం.": "2",
            "వసూళ్లు వివరం": "SHG అంతర్గత అప్పు వసూళ్లు మొత్తం",
            "వసూళ్లు మొత్తం": f"{total_shg_internal_paid:.2f}" if total_shg_internal_paid > 0 else "",
            "చెల్లింపులు వివరం": "VOకు చెల్లించిన SN పొదుపులు",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "చేతి నిల్వ",
            "బ్యాంక్ నిల్వలు విలువ": ""
        },
        {
            "క్రమ.సం.": "3",
            "వసూళ్లు వివరం": "బ్యాంక్ లోన్ వసూళ్లు మొత్తం",
            "వసూళ్లు మొత్తం": f"{total_bank_paid:.2f}" if total_bank_paid > 0 else "",
            "చెల్లింపులు వివరం": "బ్యాంక్ లోన్ ఋణం",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "పొదుపు ఖాతా",
            "బ్యాంక్ నిల్వలు విలువ": ""
        },
        {
            "క్రమ.సం.": "4",
            "వసూళ్లు వివరం": "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ అప్పు",
            "వసూళ్లు మొత్తం": f"{total_streenidhi_paid:.2f}" if total_streenidhi_paid > 0 else "",
            "చెల్లింపులు వివరం": "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ ఋణం",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "బ్యాంక్ లోన్ ఖాతా",
            "బ్యాంక్ నిల్వలు విలువ": ""
        },
        {
            "క్రమ.సం.": "5",
            "వసూళ్లు వివరం": "ఉన్నతి",
            "వసూళ్లు మొత్తం": "",
            "చెల్లింపులు వివరం": "ఉన్నతి ఋణం కు",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "ఈ నెల Bank నగదు జమ వివరాలు",
            "బ్యాంక్ నిల్వలు విలువ": ""
        },
        {
            "క్రమ.సం.": "6",
            "వసూళ్లు వివరం": "CIF అప్పు",
            "వసూళ్లు మొత్తం": "",
            "చెల్లింపులు వివరం": "CIF ఋణం కు",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "అమౌంట్ రూ.",
            "బ్యాంక్ నిల్వలు విలువ": "అక్షరాల"
        },
        {
            "క్రమ.సం.": "7",
            "వసూళ్లు వివరం": "VO అంతర్గత అప్పు",
            "వసూళ్లు మొత్తం": "",
            "చెల్లింపులు వివరం": "VO అంతర్గత ఋణం కు",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "జమచేసిన సభ్యురాలు పేరు",
            "బ్యాంక్ నిల్వలు విలువ": ""
        },
        {
            "క్రమ.సం.": "8",
            "వసూళ్లు వివరం": "SHG కు వచ్చిన కొత్త ఋణాలు",
            "వసూళ్లు మొత్తం": "",
            "చెల్లింపులు వివరం": "సభ్యులకు ఇచ్చిన కొత్త అప్పులు",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "జమచేసిన సభ్యురాలు సంతకం",
            "బ్యాంక్ నిల్వలు విలువ": ""
        },
        {
            "క్రమ.సం.": "9",
            "వసూళ్లు వివరం": "Bank వడ్డీ",
            "వసూళ్లు మొత్తం": "",
            "చెల్లింపులు వివరం": "Bank సర్వీస్ ఛార్జీలు",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "",
            "బ్యాంక్ నిల్వలు విలువ": "SHG స్టాంప్ & లీడర్స్ సంతకం"
        },
        {
            "క్రమ.సం.": "10",
            "వసూళ్లు వివరం": "ఇతర వసూళ్లు",
            "వసూళ్లు మొత్తం": "",
            "చెల్లింపులు వివరం": "VOకు చెల్లించిన సభ్యత్వరుసుము",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "",
            "బ్యాంక్ నిల్వలు విలువ": ""
        },
        {
            "క్రమ.సం.": "11",
            "వసూళ్లు వివరం": "గ్రాంట్స్ (RF/ఇతరములు)",
            "వసూళ్లు మొత్తం": "",
            "చెల్లింపులు వివరం": "ఈ నెలలో SB A/C నంద జమ చేసిన మొత్తం",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "",
            "బ్యాంక్ నిల్వలు విలువ": "VOA సంతకం"
        },
        {
            "క్రమ.సం.": "మొత్తం:",
            "వసూళ్లు వివరం": "",
            "వసూళ్లు మొత్తం": f"{total_collections:.2f}" if total_collections > 0 else "",
            "చెల్లింపులు వివరం": "మొత్తం:",
            "చెల్లింపులు మొత్తం": "",
            "బ్యాంక్ నిల్వలు వివరం": "",
            "బ్యాంక్ నిల్వలు విలువ": ""
        }
    ]
    
    reference_df = pd.DataFrame(reference_data)
    
    logger.info(f"Transformed to official format: Main={len(main_df)} rows, Reference={len(reference_df)} rows")
    logger.info(f"Totals - Savings: {total_savings_this_month:.2f}, Collections: {total_collections:.2f}")
    
    return main_df, reference_df

@app.route('/api/health', methods=['GET'])
def health_check():
    status = {
        "status": "healthy",
        "vision_api_initialized": vision_client is not None
    }
    logger.info(f"Health check: {status}")
    return jsonify(status)

@app.route('/api/extract-tables', methods=['POST'])
def extract_tables():
    temp_image_path = None
    cell_result_dir = None
    
    try:
        logger.info("="*80)
        logger.info("Received extract-tables request")
        
        if vision_client is None:
            logger.error("Vision API client not initialized")
            return jsonify({
                "success": False, 
                "error": "Vision API not configured"
            }), 500
        
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        uploaded_file = request.files['file']
        
        if uploaded_file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        logger.info(f"Processing file: {uploaded_file.filename}")
        
        # Save uploaded file temporarily
        temp_dir = 'temp_processing'
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{uploaded_file.filename}")
        uploaded_file.save(temp_image_path)
        
        # STEP 1: Use table.py to detect and crop cells
        logger.info("STEP 1: Detecting table structure and cropping cells...")
        cell_result_dir = os.path.join(temp_dir, 'cell_crops')
        
        # Process with table.py - columns 1,2,3,5,7,9
        table_result = process_table_image(
            temp_image_path, 
            cell_result_dir,
            selected_columns=[1, 2, 3, 5, 7, 9]
        )
        
        if 'error' in table_result:
            raise Exception(f"Table detection failed: {table_result['error']}")
        
        cell_results = table_result.get('results', [])
        logger.info(f"Detected {len(cell_results)} cells")
        
        if len(cell_results) == 0:
            raise Exception("No cells detected in image")
        
        # STEP 2: Process each cell with Google Vision API
        logger.info("STEP 2: Running OCR on each cell...")
        cell_ocr_data = []
        
        for cell_info in cell_results:
            # Use preprocessed (enhanced) cell image
            prep_crop_path = os.path.join(
                table_result['result_dir'], 
                cell_info['preprocessed_crop_file']
            )
            
            if not os.path.exists(prep_crop_path):
                logger.warning(f"Cell crop not found: {prep_crop_path}")
                continue
            
            # Read cell image
            with open(prep_crop_path, 'rb') as img_file:
                cell_content = img_file.read()
            
            # OCR with Vision API
            cell_image = vision.Image(content=cell_content)
            cell_context = vision.ImageContext(language_hints=['te', 'en'])
            cell_response = vision_client.document_text_detection(
                image=cell_image, 
                image_context=cell_context
            )
            
            # Extract text
            cell_text = ""
            if cell_response.text_annotations:
                cell_text = cell_response.text_annotations[0].description.strip()
            
            cell_ocr_data.append({
                'row': cell_info['display_row'],
                'col': cell_info['col_index'],
                'text': cell_text,
                'coords': cell_info['coords']
            })
        
        logger.info(f"OCR completed for {len(cell_ocr_data)} cells")
        
        # STEP 3: Transform to official format
        logger.info("STEP 3: Transforming to official SHG table format...")

        # Create initial DataFrame from OCR data
        rows_dict = defaultdict(dict)
        for cell in cell_ocr_data:
            rows_dict[cell['row']][cell['col']] = cell['text']

        sorted_rows = sorted(rows_dict.items())

        # Temporary column mapping for extracted data
        temp_column_headers = {
            1: "UID",
            2: "సభ్యురాలు పేరు",
            3: "ఈ నెల పొదుపు",
            5: "SHG అంతర్గత అప్పు - కట్టిన మొత్తం",
            7: "బ్యాంక్ అప్పు - కట్టిన మొత్తం",
            9: "స్త్రీనిధి/HD/సీడ్ క్యాపిటల్ - కట్టిన మొత్తం"
        }

        temp_data = []
        for row_idx, row_cells in sorted_rows:
            row_dict = {}
            for col_idx, header in temp_column_headers.items():
                row_dict[header] = row_cells.get(col_idx, "")
            temp_data.append(row_dict)

        temp_df = pd.DataFrame(temp_data)

        # Transform to official format (15 rows main + 12 rows reference)
        # Metadata will be passed from frontend in future enhancement
        metadata = {
            'village': '.....................',
            'meeting_date': '............'
        }

        main_df, reference_df = transform_to_official_format(temp_df, metadata)

        logger.info(f"Final main table: {len(main_df)} rows × {len(main_df.columns)} columns")
        logger.info(f"Final reference table: {len(reference_df)} rows")

        # Generate outputs
        main_csv = main_df.to_csv(index=False, encoding='utf-8-sig')
        ref_csv = reference_df.to_csv(index=False, encoding='utf-8-sig')
        combined_csv = f"### MAIN TABLE ###\n{main_csv}\n\n### REFERENCE TABLE ###\n{ref_csv}"

        json_content = {
            "main": main_df.to_dict(orient='records'),
            "reference": reference_df.to_dict(orient='records')
        }

        html_content = create_combined_html(main_df, reference_df, metadata)

        tables = [{
            "table_id": 0,
            "row_count": len(main_df),
            "col_count": len(main_df.columns),
            "dataframe": main_df.to_dict(orient='records'),
            "csv": combined_csv,
            "json": json.dumps(json_content, ensure_ascii=False),
            "html": html_content,
            "headers": list(main_df.columns),
            "has_main_table": True,
            "has_reference_table": True,
            "reference_data": reference_df.to_dict(orient='records')
        }]
        
        logger.info(f"Successfully extracted {len(tables)} table(s)")
        
        return jsonify({
            "success": True,
            "tables": tables,
            "extraction_method": "cell-by-cell-vision-api",
            "total_tables": len(tables),
            "total_rows": len(main_df),  # ✅ Fixed: Use main_df instead of df
            "total_cells_processed": len(cell_ocr_data)
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        # Cleanup temp files
        try:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if cell_result_dir and os.path.exists(cell_result_dir):
                shutil.rmtree(cell_result_dir)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

@app.route('/api/view-table-html', methods=['POST'])
def view_table_html():
    """View tables as formatted HTML - simplified version"""
    try:
        if 'file' not in request.files:
            return Response("No file uploaded", status=400)
        
        uploaded_file = request.files['file']
        
        # For now, redirect to use the main extract-tables endpoint
        return Response(
            "Please use /api/extract-tables endpoint for processing", 
            status=400
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        error_html = f'''
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body style="padding: 20px;">
            <h2 style="color: red;">Error Processing Image</h2>
            <p><strong>Error:</strong> {str(e)}</p>
            <pre style="background: #f5f5f5; padding: 15px;">{traceback.format_exc()}</pre>
        </body>
        </html>
        '''
        return Response(error_html, mimetype='text/html', status=500)

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("Cell-by-Cell OCR Table Processing System Starting...")
    print("=" * 70)
    print("Features:")
    print("  - Table structure detection via table.py")
    print("  - Individual cell cropping and enhancement")
    print("  - Google Vision API OCR on each cell")
    print("  - Accurate data reconstruction")
    print("  - Columns: 1, 2, 3, 5, 7, 9 (filtered)")
    print("=" * 70)
    print("Endpoints:")
    print("  Health: http://localhost:5000/api/health")
    print("  Extract: http://localhost:5000/api/extract-tables")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=True)