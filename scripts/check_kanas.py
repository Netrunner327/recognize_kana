import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 46 basic hiragana characters in romanized form
HIRAGANA_ROMANJI = [
    'a', 'i', 'u', 'e', 'o',  # あ い う え お
    'ka', 'ki', 'ku', 'ke', 'ko',  # か き く け こ
    'sa', 'shi', 'su', 'se', 'so',  # さ し す せ そ
    'ta', 'chi', 'tsu', 'te', 'to',  # た ち つ て と
    'na', 'ni', 'nu', 'ne', 'no',  # な に ぬ ね の
    'ha', 'hi', 'fu', 'he', 'ho',  # は ひ ふ へ ほ
    'ma', 'mi', 'mu', 'me', 'mo',  # ま み む め も
    'ya', 'yu', 'yo',  # や ゆ よ
    'ra', 'ri', 'ru', 're', 'ro',  # ら り る れ ろ
    'wa', 'wo', 'n'  # わ を ん
]

# 46 basic katakana characters in romanized form
KATAKANA_ROMANJI = [
    'A', 'I', 'U', 'E', 'O',  # ア イ ウ エ オ
    'KA', 'KI', 'KU', 'KE', 'KO',  # カ キ ク ケ コ
    'SA', 'SHI', 'SU', 'SE', 'SO',  # サ シ ス セ ソ
    'TA', 'CHI', 'TSU', 'TE', 'TO',  # タ チ ツ テ ト
    'NA', 'NI', 'NU', 'NE', 'NO',  # ナ ニ ヌ ネ ノ
    'HA', 'HI', 'FU', 'HE', 'HO',  # ハ ヒ フ ヘ ホ
    'MA', 'MI', 'MU', 'ME', 'MO',  # マ ミ ム メ モ
    'YA', 'YU', 'YO',  # ヤ ユ ヨ
    'RA', 'RI', 'RU', 'RE', 'RO',  # ラ リ ル レ ロ
    'WA', 'WO', 'N'  # ワ ヲ ン
]

# Load dataset
print("Loading dataset...")
data = np.load('data/processed/kana_dataset.npz')

hiragana_images = data['hiragana_images']
hiragana_codes = data['hiragana_codes']
katakana_images = data['katakana_images']
katakana_codes = data['katakana_codes']

print(f"\nDataset Summary:")
print(f"  Hiragana: {len(hiragana_images):,} samples, {len(np.unique(hiragana_codes))} unique codes")
print(f"  Katakana: {len(katakana_images):,} samples, {len(np.unique(katakana_codes))} unique codes")

# Check distribution
from collections import Counter
hira_dist = Counter(hiragana_codes)
kata_dist = Counter(katakana_codes)

print(f"\n  Hiragana samples per code: min={min(hira_dist.values())}, max={max(hira_dist.values())}")
print(f"  Katakana samples per code: min={min(kata_dist.values())}, max={max(kata_dist.values())}")

# Create verification figure for HIRAGANA
print("\nCreating hiragana verification...")
fig, axes = plt.subplots(8, 6, figsize=(18, 24))
fig.suptitle('Hiragana Dataset Verification (Code 0-45)', fontsize=20, y=0.995)

for idx in range(46):
    row = idx // 6
    col = idx % 6
    
    # Find first sample with this code
    samples = hiragana_images[hiragana_codes == idx]
    
    if len(samples) > 0:
        axes[row, col].imshow(samples[0], cmap='gray', vmin=0, vmax=1)
        title = f'{HIRAGANA_ROMANJI[idx]} (Code {idx})\n{len(samples)} samples'
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    else:
        axes[row, col].text(0.5, 0.5, f'Missing\nCode {idx}', 
                           ha='center', va='center', fontsize=12, color='red')
        axes[row, col].axis('off')

# Hide extra subplots
for idx in range(46, 48):
    row = idx // 6
    col = idx % 6
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('data/processed/hiragana_verification.png', dpi=150, bbox_inches='tight')
print("✓ Saved hiragana_verification.png")

# Create verification figure for KATAKANA
print("Creating katakana verification...")
fig, axes = plt.subplots(8, 6, figsize=(18, 24))
fig.suptitle('Katakana Dataset Verification (Code 0-45)', fontsize=20, y=0.995)

for idx in range(46):
    row = idx // 6
    col = idx % 6
    
    # Find first sample with this code
    samples = katakana_images[katakana_codes == idx]
    
    if len(samples) > 0:
        axes[row, col].imshow(samples[0], cmap='gray', vmin=0, vmax=1)
        title = f'{KATAKANA_ROMANJI[idx]} (Code {idx})\n{len(samples)} samples'
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    else:
        axes[row, col].text(0.5, 0.5, f'Missing\nCode {idx}', 
                           ha='center', va='center', fontsize=12, color='red')
        axes[row, col].axis('off')

# Hide extra subplots
for idx in range(46, 48):
    row = idx // 6
    col = idx % 6
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('data/processed/katakana_verification.png', dpi=150, bbox_inches='tight')
print("✓ Saved katakana_verification.png")

# Create side-by-side comparison
print("Creating side-by-side comparison...")
fig, axes = plt.subplots(8, 12, figsize=(36, 24))
fig.suptitle('Hiragana vs Katakana Comparison (Code 0-45)', fontsize=24, y=0.995)

for idx in range(46):
    row = idx // 6
    hira_col = (idx % 6) * 2
    kata_col = hira_col + 1
    
    # Hiragana
    hira_samples = hiragana_images[hiragana_codes == idx]
    if len(hira_samples) > 0:
        axes[row, hira_col].imshow(hira_samples[0], cmap='gray', vmin=0, vmax=1)
        axes[row, hira_col].set_title(f'{HIRAGANA_ROMANJI[idx]}\n({len(hira_samples)})', fontsize=9)
        axes[row, hira_col].axis('off')
    else:
        axes[row, hira_col].text(0.5, 0.5, 'Missing', ha='center', va='center', color='red')
        axes[row, hira_col].axis('off')
    
    # Katakana
    kata_samples = katakana_images[katakana_codes == idx]
    if len(kata_samples) > 0:
        axes[row, kata_col].imshow(kata_samples[0], cmap='gray', vmin=0, vmax=1)
        axes[row, kata_col].set_title(f'{KATAKANA_ROMANJI[idx]}\n({len(kata_samples)})', fontsize=9)
        axes[row, kata_col].axis('off')
    else:
        axes[row, kata_col].text(0.5, 0.5, 'Missing', ha='center', va='center', color='red')
        axes[row, kata_col].axis('off')

# Hide extra subplots
for idx in range(46, 48):
    row = idx // 6
    for col in [(idx % 6) * 2, (idx % 6) * 2 + 1]:
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('data/processed/kana_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved kana_comparison.png")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
print("\nGenerated files:")
print("  1. hiragana_verification.png - All 46 hiragana characters")
print("  2. katakana_verification.png - All 46 katakana characters")
print("  3. kana_comparison.png - Side-by-side comparison")
print("\nCheck these images to verify:")
print("  - All characters are present (codes 0-45)")
print("  - Characters match expected hiragana/katakana")
print("  - No kanji or other characters mixed in")
print("  - Each code has 1000 samples")