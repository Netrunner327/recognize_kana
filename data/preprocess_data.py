import numpy as np
from PIL import Image
import os
import struct
from scipy.ndimage import rotate, shift, zoom as scipy_zoom
import random
from collections import defaultdict

def normalize_hiragana_code(jis_code):
    """
    Normalize ETL8G JIS code to base hiragana index (0-45)
    
    ETL8G uses JIS X 0208:
    - Basic hiragana: 0x2422-0x2473 (あ-ん)
    """
    HIRAGANA_START = 0x2422  # あ
    
    if 0x2422 <= jis_code <= 0x2473:
        normalized = jis_code - HIRAGANA_START
        return normalized % 46
    else:
        return jis_code % 46


def read_etl8g(filename, debug=False):
    """
    Read ETL8G dataset (hiragana ONLY - 46 basic characters)
    Record size: 8199 bytes
    Image format: 128x127, 4-bit grayscale (packed nibbles)
    Image data: bytes 61-8188 (8128 bytes)
    """
    from collections import Counter
    
    # Define the 46 basic hiragana JIS X 0208 codes in gojūon order
    # Based on JIS X 0208 row 04 (hiragana)
    BASIC_HIRAGANA_CODES = {
        0x2422,  # あ (a)
        0x2424,  # い (i)
        0x2426,  # う (u)
        0x2428,  # え (e)
        0x242A,  # お (o)
        0x242B,  # か (ka)
        0x242D,  # き (ki)
        0x242F,  # く (ku)
        0x2431,  # け (ke)
        0x2433,  # こ (ko)
        0x2435,  # さ (sa)
        0x2437,  # し (shi)
        0x2439,  # す (su)
        0x243B,  # せ (se)
        0x243D,  # そ (so)
        0x243F,  # た (ta)
        0x2441,  # ち (chi)
        0x2443,  # つ (tsu)
        0x2445,  # て (te)
        0x2447,  # と (to)
        0x2448,  # な (na)
        0x244B,  # に (ni)
        0x244C,  # ぬ (nu)
        0x244D,  # ね (ne)
        0x244E,  # の (no)
        0x244F,  # は (ha)
        0x2452,  # ひ (hi)
        0x2454,  # ふ (fu)
        0x2456,  # へ (he)
        0x2458,  # ほ (ho)
        0x245E,  # ま (ma)
        0x245F,  # み (mi)
        0x2460,  # む (mu)
        0x2461,  # め (me)
        0x2462,  # も (mo)
        0x2463,  # や (ya)
        0x2465,  # ゆ (yu)
        0x2467,  # よ (yo)
        0x2469,  # ら (ra)
        0x246A,  # り (ri)
        0x246B,  # る (ru)
        0x246C,  # れ (re)
        0x246D,  # ろ (ro)
        0x246F,  # わ (wa)
        0x2472,  # を (wo)
        0x2473,  # ん (n)
    }
    
    # Create mapping from JIS code to gojūon index (0-45)
    HIRAGANA_CODE_TO_INDEX = {code: idx for idx, code in enumerate(sorted(BASIC_HIRAGANA_CODES))}
    
    with open(filename, 'rb') as f:
        records = []
        skipped = 0
        all_codes = []
        
        while True:
            s = f.read(8199)
            if len(s) < 8199:
                break
            
            # Extract JIS code (bytes 3-4, 1-indexed) - Big endian
            code = struct.unpack('>H', s[2:4])[0]
            all_codes.append(code)
            
            # FILTER: Only accept the 46 basic hiragana codes
            if code not in BASIC_HIRAGANA_CODES:
                skipped += 1
                continue
            
            # Map to gojūon index (0-45)
            normalized_code = HIRAGANA_CODE_TO_INDEX[code]
            
            # Image data: bytes 61-8188 (0-indexed: 60-8188)
            img_bytes = s[60:60+8128]
            
            # Unpack 4-bit data to 8-bit
            img_array = np.zeros((127, 128), dtype=np.uint8)
            
            for byte_idx in range(len(img_bytes)):
                byte_val = img_bytes[byte_idx]
                
                pixel1 = (byte_val >> 4) & 0x0F
                pixel2 = byte_val & 0x0F
                
                pixel_idx = byte_idx * 2
                
                row1 = pixel_idx // 128
                col1 = pixel_idx % 128
                if row1 < 127:
                    img_array[row1, col1] = (15 - pixel1) * 17
                
                pixel_idx += 1
                row2 = pixel_idx // 128
                col2 = pixel_idx % 128
                if row2 < 127:
                    img_array[row2, col2] = (15 - pixel2) * 17
            
            # Resize to 64x64
            img = Image.fromarray(img_array, mode='L')
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
            records.append({
                'code': normalized_code,
                'original_code': code,
                'image': np.array(img),
                'type': 'hiragana'
            })
        
        # Debug: Show what codes we found
        if debug and all_codes:
            code_counts = Counter(all_codes)
            hiragana_codes = [c for c in code_counts.keys() if c in BASIC_HIRAGANA_CODES]
            print(f"    Found {len(hiragana_codes)} unique hiragana codes: {len(records)} samples")
            print(f"    Code range in file: 0x{min(all_codes):04X}-0x{max(all_codes):04X}")
            if len(hiragana_codes) < 46:
                print(f"    WARNING: Only {len(hiragana_codes)} hiragana codes (expected 46)")
        
        if skipped > 0:
            print(f"    Skipped {skipped} non-basic hiragana characters")
    
    return records


def read_etl1(filename):
    """
    Read ETL1 dataset (katakana)
    Record size: 2052 bytes
    Image format: 64x63, 4-bit grayscale (packed nibbles)
    Image data: bytes 33-2048 (2016 bytes)
    """
    with open(filename, 'rb') as f:
        records = []
        while True:
            s = f.read(2052)
            if len(s) < 2052:
                break
            
            # Extract ASCII character code (bytes 3-4, 1-indexed)
            # For katakana files (07-13), this contains katakana codes
            char_code = s[2:4].decode('ascii', errors='ignore').strip()
            
            # Extract JIS code (byte 7, 1-indexed = byte 6 in 0-indexed)
            jis_code = s[6]
            
            # Image data: bytes 33-2048 (0-indexed: 32-2048)
            img_bytes = s[32:32+2016]
            
            # Unpack 4-bit data to 8-bit
            img_array = np.zeros((63, 64), dtype=np.uint8)
            
            for byte_idx in range(len(img_bytes)):
                byte_val = img_bytes[byte_idx]
                
                pixel1 = (byte_val >> 4) & 0x0F
                pixel2 = byte_val & 0x0F
                
                pixel_idx = byte_idx * 2
                
                row1 = pixel_idx // 64
                col1 = pixel_idx % 64
                if row1 < 63:
                    img_array[row1, col1] = (15 - pixel1) * 17
                
                pixel_idx += 1
                row2 = pixel_idx // 64
                col2 = pixel_idx % 64
                if row2 < 63:
                    img_array[row2, col2] = (15 - pixel2) * 17
            
            # Resize to 64x64
            img = Image.fromarray(img_array, mode='L')
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
            records.append({
                'code': jis_code,
                'char_code': char_code,  # For debugging
                'image': np.array(img),
                'type': 'katakana'
            })
    
    return records


def augment_image(img_array):
    """Create augmented version of image"""
    img = img_array.copy().astype(np.float32)
    
    angle = random.uniform(-12, 12)
    img = rotate(img, angle, reshape=False, cval=255, order=1)
    
    shift_x = random.randint(-3, 3)
    shift_y = random.randint(-3, 3)
    img = shift(img, [shift_y, shift_x], cval=255, order=1)
    
    zoom_factor = random.uniform(0.92, 1.08)
    img = scipy_zoom(img, zoom_factor, order=1)
    
    h, w = img.shape
    if h > 64 or w > 64:
        start_h = max(0, (h - 64) // 2)
        start_w = max(0, (w - 64) // 2)
        img = img[start_h:start_h+64, start_w:start_w+64]
    
    if img.shape[0] < 64 or img.shape[1] < 64:
        pad_h = max(0, 64 - img.shape[0])
        pad_w = max(0, 64 - img.shape[1])
        img = np.pad(img, 
                    ((pad_h//2, (pad_h+1)//2), (pad_w//2, (pad_w+1)//2)), 
                    mode='constant', 
                    constant_values=255)[:64, :64]
    
    noise = np.random.normal(0, 3, img.shape)
    img = np.clip(img + noise, 0, 255)
    
    return img.astype(np.uint8)


def balance_with_augmentation(hiragana_data, katakana_data, 
                              target_samples_per_char=1000,
                              katakana_samples_per_char=25,
                              katakana_augmentation_factor=39):
    """Balance dataset with augmentation for BOTH hiragana and katakana"""
    print("\n" + "="*60)
    print("BALANCING DATASET WITH AUGMENTATION")
    print("="*60)
    
    # Group hiragana by code
    hiragana_by_code = defaultdict(list)
    for record in hiragana_data:
        hiragana_by_code[record['code']].append(record)
    
    # Augment hiragana to reach target
    balanced_hiragana = []
    print(f"\n1. Augmenting Hiragana:")
    print(f"   Target: {target_samples_per_char} samples per character")
    print(f"   Found {len(hiragana_by_code)} unique characters")
    print(f"   Current avg: {len(hiragana_data) / len(hiragana_by_code):.0f} samples/char")
    
    for code, records in sorted(hiragana_by_code.items()):
        originals_count = len(records)
        if originals_count == 0:
            continue
            
        if originals_count >= target_samples_per_char:
            sampled = random.sample(records, target_samples_per_char)
            balanced_hiragana.extend(sampled)
        else:
            augmentations_needed = target_samples_per_char - originals_count
            augmentations_per_original = augmentations_needed // originals_count
            extra_augmentations = augmentations_needed % originals_count
            
            balanced_hiragana.extend(records)
            
            for idx, record in enumerate(records):
                num_augs = augmentations_per_original + (1 if idx < extra_augmentations else 0)
                for _ in range(num_augs):
                    aug_img = augment_image(record['image'])
                    balanced_hiragana.append({
                        'code': record['code'],
                        'image': aug_img,
                        'type': 'hiragana'
                    })
    
    print(f"   Result: {len(hiragana_data)} → {len(balanced_hiragana)} samples")
    print(f"   ({len(hiragana_by_code)} chars × {len(balanced_hiragana)//max(len(hiragana_by_code), 1)} samples/char)")
    
    # Group katakana by character code (NOW IN HEPBURN romanization after normalization)
    katakana_by_char = defaultdict(list)
    for record in katakana_data:
        char_code = record.get('char_code', '').strip().upper()
        katakana_by_char[char_code].append(record)
    
    # Define the correct order of 46 katakana in Hepburn romanization (matching hiragana order)
    KATAKANA_ORDER = [
        'A', 'I', 'U', 'E', 'O',           # ア イ ウ エ オ (0-4)
        'KA', 'KI', 'KU', 'KE', 'KO',      # カ キ ク ケ コ (5-9)
        'SA', 'SHI', 'SU', 'SE', 'SO',     # サ シ ス セ ソ (10-14) - SHI not SI
        'TA', 'CHI', 'TSU', 'TE', 'TO',    # タ チ ツ テ ト (15-19) - CHI not TI, TSU not TU
        'NA', 'NI', 'NU', 'NE', 'NO',      # ナ ニ ヌ ネ ノ (20-24)
        'HA', 'HI', 'FU', 'HE', 'HO',      # ハ ヒ フ ヘ ホ (25-29) - FU not HU
        'MA', 'MI', 'MU', 'ME', 'MO',      # マ ミ ム メ モ (30-34)
        'YA', 'YU', 'YO',                  # ヤ ユ ヨ (35-37)
        'RA', 'RI', 'RU', 'RE', 'RO',      # ラ リ ル レ ロ (38-42)
        'WA', 'WO', 'N'                    # ワ ヲ ン (43-45)
    ]
    
    char_to_index = {char: idx for idx, char in enumerate(KATAKANA_ORDER)}
    
    print(f"\n   Mapping katakana to indices 0-45 in gojūon order")
    
    # Augment katakana
    augmented_katakana = []
    print(f"\n2. Augmenting Katakana:")
    print(f"   Limiting to {katakana_samples_per_char} originals per character")
    print(f"   Creating {katakana_augmentation_factor} augmented versions per original...")
    
    total_originals = 0
    missing_chars = []
    
    for char_code in KATAKANA_ORDER:
        records = katakana_by_char.get(char_code, [])
        
        if not records:
            missing_chars.append(char_code)
            print(f"   WARNING: No samples found for '{char_code}'")
            continue
        
        # Limit originals per character
        sample_size = min(len(records), katakana_samples_per_char)
        sampled = random.sample(records, sample_size)
        total_originals += sample_size
        
        for record in sampled:
            # Map to 0-45 index based on gojūon order
            record['code'] = char_to_index[char_code]
            augmented_katakana.append(record)
            
            # Create augmented versions
            for _ in range(katakana_augmentation_factor):
                aug_img = augment_image(record['image'])
                augmented_katakana.append({
                    'code': char_to_index[char_code],
                    'image': aug_img,
                    'type': 'katakana'
                })
    
    if missing_chars:
        print(f"\n   ERROR: Missing {len(missing_chars)} katakana: {missing_chars}")
    
    print(f"   Result: {total_originals} originals → {len(augmented_katakana)} samples")
    if total_originals > 0:
        print(f"   ({len(KATAKANA_ORDER) - len(missing_chars)} chars × {len(augmented_katakana)//(len(KATAKANA_ORDER) - len(missing_chars))} samples/char)")
    
    print(f"\n3. Final Balanced Dataset:")
    print(f"   Hiragana: {len(balanced_hiragana):,} samples")
    print(f"   Katakana: {len(augmented_katakana):,} samples")
    print(f"   Total:    {len(balanced_hiragana) + len(augmented_katakana):,} samples")
    
    if len(balanced_hiragana) > 0 and len(augmented_katakana) > 0:
        ratio = len(balanced_hiragana) / len(augmented_katakana)
        print(f"   Ratio:    {ratio:.2f}:1", "✓ (balanced!)" if 0.9 <= ratio <= 1.1 else "⚠ (imbalanced)")
    print("="*60 + "\n")
    
    return balanced_hiragana, augmented_katakana


def filter_katakana(records, debug=False):
    """
    Filter only katakana characters from ETL1
    
    ETL1C files 07-13 should contain katakana, but we need to verify
    by checking both JIS code AND character code (ASCII representation)
    """
    if debug:
        from collections import Counter
        unique_codes = sorted(set(record['code'] for record in records))
        print(f"\nDEBUG: Found {len(unique_codes)} unique JIS codes in ETL1")
        print(f"Code range: 0x{min(unique_codes):02X}-0x{max(unique_codes):02X}")
        
        # Show what's at each code
        print(f"\nDetailed code mapping:")
        code_to_char = {}
        for record in records:
            if record['code'] not in code_to_char:
                code_to_char[record['code']] = record.get('char_code', '')
        
        for code in sorted(code_to_char.keys())[:20]:
            char = code_to_char[code]
            print(f"  JIS 0x{code:02X} = '{char}'")
    
    # Define valid katakana character codes (romanized)
    # ETL1 uses Kunrei-shiki romanization (TI, TU, SI, HU, ZI, ZU, etc.)
    # We accept these and convert to Hepburn later
    VALID_KATAKANA_CHARS = {
        'A', 'I', 'U', 'E', 'O',           # ア イ ウ エ オ
        'KA', 'KI', 'KU', 'KE', 'KO',      # カ キ ク ケ コ
        'SA', 'SI', 'SU', 'SE', 'SO',      # サ シ ス セ ソ (SI = shi)
        'TA', 'TI', 'TU', 'TE', 'TO',      # タ チ ツ テ ト (TI = chi, TU = tsu)
        'NA', 'NI', 'NU', 'NE', 'NO',      # ナ ニ ヌ ネ ノ
        'HA', 'HI', 'HU', 'HE', 'HO',      # ハ ヒ フ ヘ ホ (HU = fu)
        'MA', 'MI', 'MU', 'ME', 'MO',      # マ ミ ム メ モ
        'YA', 'YU', 'YO',                  # ヤ ユ ヨ
        'RA', 'RI', 'RU', 'RE', 'RO',      # ラ リ ル レ ロ
        'WA', 'WO', 'N'                    # ワ ヲ ン
    }
    
    # Normalize Kunrei-shiki to Hepburn romanization
    KUNREI_TO_HEPBURN = {
        'SI': 'SHI',   # し/シ
        'TI': 'CHI',   # ち/チ
        'TU': 'TSU',   # つ/ツ
        'HU': 'FU',    # ふ/フ
    }
    
    # Filter records and normalize romanization
    katakana_records = []
    rejected = []
    
    for record in records:
        char_code = record.get('char_code', '').strip().upper()
        
        if char_code in VALID_KATAKANA_CHARS:
            # Convert Kunrei-shiki to Hepburn
            normalized_char = KUNREI_TO_HEPBURN.get(char_code, char_code)
            record['char_code'] = normalized_char  # Update to Hepburn
            katakana_records.append(record)
        else:
            rejected.append((record['code'], char_code))
    
    if debug:
        print(f"\nFiltered {len(katakana_records)} katakana from {len(records)} total records")
        print(f"Rejected {len(rejected)} records")
        
        if rejected:
            from collections import Counter
            rejected_counts = Counter(rejected)
            print(f"\nRejected characters (showing first 10):")
            for (code, char), count in list(rejected_counts.items())[:10]:
                print(f"  JIS 0x{code:02X} = '{char}': {count} samples")
        
        # Verify we got 46 unique katakana
        unique_chars = set(r.get('char_code', '').strip().upper() for r in katakana_records)
        print(f"\nKept {len(unique_chars)} unique katakana characters:")
        print(f"  {sorted(unique_chars)}")
    
    return katakana_records


def find_etl_files(base_path, prefix):
    """Recursively find ETL files"""
    import re
    files = []
    if not os.path.exists(base_path):
        return files
    
    for root, dirs, filenames in os.walk(base_path):
        for filename in filenames:
            if prefix == 'ETL1':
                # Only katakana files: etl1c_07 to etl1c_13
                upper = filename.upper()
                if 'ETL1C' in upper or 'ETL1' in upper:
                    numbers = re.findall(r'\d+', filename)
                    if numbers:
                        file_num = int(numbers[-1])
                        if 7 <= file_num <= 13:
                            filepath = os.path.join(root, filename)
                            if os.path.isfile(filepath) and not filename.endswith('.zip') and 'INFO' not in upper:
                                files.append(filepath)
                                print(f"  Including: {filename}")
            else:
                # ETL8G: include all files
                if filename.upper().startswith(prefix.upper()) and not filename.endswith('.zip') and 'INFO' not in filename.upper():
                    filepath = os.path.join(root, filename)
                    if os.path.isfile(filepath):
                        files.append(filepath)
    
    return sorted(files)


def load_all_kana(etl8g_path, etl1_path, debug=False, balance=True):
    """Load both hiragana and katakana datasets"""
    hiragana_data = []
    katakana_data = []
    
    # Load ETL8G (hiragana)
    etl8g_files = find_etl_files(etl8g_path, 'ETL8G')
    print(f"Found {len(etl8g_files)} ETL8G files")
    
    for filepath in etl8g_files:
        print(f"Loading {os.path.basename(filepath)}...")
        hiragana_data.extend(read_etl8g(filepath, debug=True))  # Enable debug
    
    # Load ETL1 (katakana)
    etl1_files = find_etl_files(etl1_path, 'ETL1')
    print(f"\nFound {len(etl1_files)} ETL1 files")
    
    all_etl1_records = []
    for filepath in etl1_files:
        print(f"Loading {os.path.basename(filepath)}...")
        all_etl1_records.extend(read_etl1(filepath))
    
    katakana_data = filter_katakana(all_etl1_records, debug=debug)
    
    print(f"\nInitial load:")
    print(f"  Hiragana: {len(hiragana_data)} samples")
    print(f"  Katakana: {len(katakana_data)} samples")
    
    # Debug: Check distribution
    if debug:
        from collections import Counter
        hira_code_counts = Counter(r['code'] for r in hiragana_data)
        print(f"\n  Hiragana distribution:")
        print(f"    Unique codes: {len(hira_code_counts)}")
        print(f"    Min samples per code: {min(hira_code_counts.values()) if hira_code_counts else 0}")
        print(f"    Max samples per code: {max(hira_code_counts.values()) if hira_code_counts else 0}")
        print(f"    Avg samples per code: {len(hiragana_data) / len(hira_code_counts) if hira_code_counts else 0:.0f}")
    
    # Balance
    if balance:
        hiragana_data, katakana_data = balance_with_augmentation(hiragana_data, katakana_data)
    
    # Normalize to [0, 1]
    print("Normalizing images to [0, 1] range...")
    for record in hiragana_data + katakana_data:
        record['image'] = record['image'].astype(np.float32) / 255.0
    
    return {'hiragana': hiragana_data, 'katakana': katakana_data}


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    os.makedirs('data/processed', exist_ok=True)
    
    print("Starting dataset extraction and preprocessing...\n")
    
    data = load_all_kana('data/raw/etl8g/', 'data/raw/etl1/', debug=True, balance=True)
    
    total_samples = len(data['hiragana']) + len(data['katakana'])
    print(f"\nFinal dataset ready: {total_samples:,} total samples")
    
    if len(data['katakana']) > 0:
        print("\nSaving balanced dataset to data/processed/kana_dataset.npz...")
        np.savez_compressed('data/processed/kana_dataset.npz',
                            hiragana_images=np.array([r['image'] for r in data['hiragana']]),
                            hiragana_codes=np.array([r['code'] for r in data['hiragana']]),
                            katakana_images=np.array([r['image'] for r in data['katakana']]),
                            katakana_codes=np.array([r['code'] for r in data['katakana']]))
        print("✓ Dataset saved successfully!")
        print(f"\nFile size: ~{os.path.getsize('data/processed/kana_dataset.npz') / (1024*1024):.1f} MB")
    else:
        print("\n✗ ERROR: No katakana samples found!")