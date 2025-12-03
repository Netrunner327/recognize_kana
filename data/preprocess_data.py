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
    Read ETL8G dataset (hiragana ONLY)
    Record size: 8199 bytes
    Image format: 128x127, 4-bit grayscale (packed nibbles)
    Image data: bytes 61-8188 (8128 bytes)
    
    Only loads hiragana characters (JIS X 0208: 0x2422-0x2473)
    """
    from collections import Counter
    
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
            
            # FILTER: Only accept hiragana codes (0x2422-0x2473)
            # This is the basic hiragana range: あ-ん (46 characters)
            if not (0x2422 <= code <= 0x2473):
                skipped += 1
                continue
            
            normalized_code = normalize_hiragana_code(code)
            
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
            hiragana_codes = [c for c in code_counts.keys() if 0x2422 <= c <= 0x2473]
            print(f"    Found {len(hiragana_codes)} unique hiragana codes: {len(records)} samples")
            print(f"    Code range in file: 0x{min(all_codes):04X}-0x{max(all_codes):04X}")
            if len(hiragana_codes) < 46:
                print(f"    WARNING: Only {len(hiragana_codes)} hiragana codes (expected 46)")
        
        if skipped > 0:
            print(f"    Skipped {skipped} non-hiragana characters")
    
    return records


def read_etl1(filename):
    """
    Read ETL1 dataset (katakana and some kanji)
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
        # Calculate how many augmentations needed per original
        originals_count = len(records)
        if originals_count == 0:
            continue
            
        # If we have enough originals, just sample
        if originals_count >= target_samples_per_char:
            sampled = random.sample(records, target_samples_per_char)
            balanced_hiragana.extend(sampled)
        else:
            # Need to augment: use all originals + create augmented versions
            augmentations_needed = target_samples_per_char - originals_count
            augmentations_per_original = augmentations_needed // originals_count
            extra_augmentations = augmentations_needed % originals_count
            
            # Add all originals
            balanced_hiragana.extend(records)
            
            # Create augmented versions
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
    
    # Group katakana by code
    katakana_by_code = defaultdict(list)
    for record in katakana_data:
        katakana_by_code[record['code']].append(record)
    
    # Map first 46 katakana codes to 0-45 indices
    katakana_jis_codes = sorted(katakana_by_code.keys())[:46]
    code_to_index = {code: idx for idx, code in enumerate(katakana_jis_codes)}
    
    print(f"\n   Using 46 katakana codes (0x{katakana_jis_codes[0]:02X}-0x{katakana_jis_codes[-1]:02X})")
    
    # Augment katakana
    augmented_katakana = []
    print(f"\n2. Augmenting Katakana:")
    print(f"   Limiting to {katakana_samples_per_char} originals per character")
    print(f"   Creating {katakana_augmentation_factor} augmented versions per original...")
    
    total_originals = 0
    for code in katakana_jis_codes:
        records = katakana_by_code.get(code, [])
        
        # Limit originals per character
        sample_size = min(len(records), katakana_samples_per_char)
        sampled = random.sample(records, sample_size)
        total_originals += sample_size
        
        for record in sampled:
            # Map to 0-45 index
            record['code'] = code_to_index[code]
            augmented_katakana.append(record)
            
            # Create augmented versions
            for _ in range(katakana_augmentation_factor):
                aug_img = augment_image(record['image'])
                augmented_katakana.append({
                    'code': code_to_index[code],
                    'image': aug_img,
                    'type': 'katakana'
                })
    
    print(f"   Result: {total_originals} originals → {len(augmented_katakana)} samples")
    print(f"   (46 chars × {len(augmented_katakana)//46} samples/char)")
    
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
    """Filter only katakana characters from ETL1 (JIS X 0201: 0xA1-0xDF)"""
    if debug:
        from collections import Counter
        unique_codes = sorted(set(record['code'] for record in records))
        print(f"\nDEBUG: Found {len(unique_codes)} unique JIS codes in ETL1")
        print(f"Code range: 0x{min(unique_codes):02X}-0x{max(unique_codes):02X}")
        
        code_counts = Counter(record['code'] for record in records)
        katakana_codes = [c for c in unique_codes if 0xA1 <= c <= 0xDF]
        print(f"Katakana codes (0xA1-0xDF): {len(katakana_codes)} found")
        for code in katakana_codes[:10]:
            print(f"  Code 0x{code:02X}: {code_counts[code]} samples")
    
    katakana_records = [r for r in records if 0xA1 <= r['code'] <= 0xDF]
    
    if debug:
        print(f"\nFiltered {len(katakana_records)} katakana from {len(records)} total records")
    
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