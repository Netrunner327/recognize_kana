import numpy as np
import matplotlib.pyplot as plt
import random

# Hiragana and Katakana character mappings
HIRAGANA = [
    'あ', 'い', 'う', 'え', 'お',
    'か', 'き', 'く', 'け', 'こ',
    'さ', 'し', 'す', 'せ', 'そ',
    'た', 'ち', 'つ', 'て', 'と',
    'な', 'に', 'ぬ', 'ね', 'の',
    'は', 'ひ', 'ふ', 'へ', 'ほ',
    'ま', 'み', 'む', 'め', 'も',
    'や', 'ゆ', 'よ',
    'ら', 'り', 'る', 'れ', 'ろ',
    'わ', 'を', 'ん'
]

KATAKANA = [
    'ア', 'イ', 'ウ', 'エ', 'オ',
    'カ', 'キ', 'ク', 'ケ', 'コ',
    'サ', 'シ', 'ス', 'セ', 'ソ',
    'タ', 'チ', 'ツ', 'テ', 'ト',
    'ナ', 'ニ', 'ヌ', 'ネ', 'ノ',
    'ハ', 'ヒ', 'フ', 'ヘ', 'ホ',
    'マ', 'ミ', 'ム', 'メ', 'モ',
    'ヤ', 'ユ', 'ヨ',
    'ラ', 'リ', 'ル', 'レ', 'ロ',
    'ワ', 'ヲ', 'ン'
]


def visualize_random_samples(dataset_path='data/processed/kana_dataset.npz', 
                             num_samples=20,
                             save_path='data/processed/dataset_visualization.png'):
    """
    Visualize random samples from the dataset
    
    Args:
        dataset_path: Path to the .npz dataset file
        num_samples: Number of random samples to display
        save_path: Where to save the visualization
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    
    hiragana_images = data['hiragana_images']
    hiragana_codes = data['hiragana_codes']
    katakana_images = data['katakana_images']
    katakana_codes = data['katakana_codes']
    
    print(f"Dataset loaded:")
    print(f"  Hiragana: {len(hiragana_images):,} samples")
    print(f"  Katakana: {len(katakana_images):,} samples")
    print(f"  Total: {len(hiragana_images) + len(katakana_images):,} samples\n")
    
    # Create figure
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    fig.suptitle('Random Dataset Samples (10 Hiragana + 10 Katakana)', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Select random hiragana samples
    hiragana_indices = random.sample(range(len(hiragana_images)), num_samples // 2)
    
    # Select random katakana samples
    katakana_indices = random.sample(range(len(katakana_images)), num_samples // 2)
    
    # Plot hiragana samples (first 10)
    for i, idx in enumerate(hiragana_indices):
        img = hiragana_images[idx]
        code = hiragana_codes[idx]
        
        # Map code to hiragana character
        # ETL8G uses JIS X 0208 codes, need to map to hiragana
        # For simplicity, use modulo to get character index
        char_idx = code % len(HIRAGANA)
        char = HIRAGANA[char_idx] if char_idx < len(HIRAGANA) else '?'
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Hiragana: {char}\nCode: {code}', 
                         fontsize=10, color='blue')
        axes[i].axis('off')
    
    # Plot katakana samples (last 10)
    for i, idx in enumerate(katakana_indices):
        img = katakana_images[idx]
        code = katakana_codes[idx]
        
        # ETL1 uses sequential codes 1-46 for katakana
        char_idx = (code - 1) % len(KATAKANA)
        char = KATAKANA[char_idx] if 0 <= char_idx < len(KATAKANA) else '?'
        
        axes[i + num_samples // 2].imshow(img, cmap='gray')
        axes[i + num_samples // 2].set_title(f'Katakana: {char}\nCode: {code}', 
                                             fontsize=10, color='red')
        axes[i + num_samples // 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.show()


def visualize_augmentation_comparison(dataset_path='data/processed/kana_dataset.npz',
                                     char_code=1,
                                     num_variations=10,
                                     save_path='data/processed/augmentation_comparison.png'):
    """
    Show multiple variations of the same character to see augmentation quality
    
    Args:
        dataset_path: Path to the .npz dataset file
        char_code: Character code to visualize (1-46 for katakana)
        num_variations: Number of variations to show
        save_path: Where to save the visualization
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    
    katakana_images = data['katakana_images']
    katakana_codes = data['katakana_codes']
    
    # Find all samples with the specified code
    matching_indices = np.where(katakana_codes == char_code)[0]
    
    if len(matching_indices) == 0:
        print(f"No samples found for character code {char_code}")
        return
    
    print(f"Found {len(matching_indices)} samples for code {char_code}")
    
    # Get character name
    char_idx = (char_code - 1) % len(KATAKANA)
    char = KATAKANA[char_idx] if 0 <= char_idx < len(KATAKANA) else '?'
    
    # Select random variations
    selected_indices = random.sample(list(matching_indices), 
                                    min(num_variations, len(matching_indices)))
    
    # Create figure
    cols = 5
    rows = (num_variations + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle(f'Augmentation Variations: {char} (Code {char_code})', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Plot variations
    for i, idx in enumerate(selected_indices):
        img = katakana_images[idx]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Variation {i+1}', fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(selected_indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Augmentation comparison saved to {save_path}")
    plt.show()


def visualize_class_distribution(dataset_path='data/processed/kana_dataset.npz',
                                 save_path='data/processed/class_distribution.png'):
    """
    Show distribution of samples across different character classes
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    
    hiragana_codes = data['hiragana_codes']
    katakana_codes = data['katakana_codes']
    
    # Count samples per character
    from collections import Counter
    hiragana_counts = Counter(hiragana_codes)
    katakana_counts = Counter(katakana_codes)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Class Distribution: Samples per Character', 
                 fontsize=16, fontweight='bold')
    
    # Hiragana distribution
    codes_h = sorted(hiragana_counts.keys())
    counts_h = [hiragana_counts[c] for c in codes_h]
    ax1.bar(range(len(codes_h)), counts_h, color='blue', alpha=0.7)
    ax1.set_title('Hiragana Distribution', fontsize=14)
    ax1.set_xlabel('Character Index')
    ax1.set_ylabel('Number of Samples')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=1000, color='r', linestyle='--', label='Target: 1000 samples/char')
    ax1.legend()
    
    # Katakana distribution
    codes_k = sorted(katakana_counts.keys())
    counts_k = [katakana_counts[c] for c in codes_k]
    ax2.bar(range(len(codes_k)), counts_k, color='red', alpha=0.7)
    ax2.set_title('Katakana Distribution (Augmented)', fontsize=14)
    ax2.set_xlabel('Character Index')
    ax2.set_ylabel('Number of Samples')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=1000, color='r', linestyle='--', label='Target: 1000 samples/char')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Class distribution saved to {save_path}")
    plt.show()
    
    # Print statistics
    print(f"\nHiragana Statistics:")
    print(f"  Characters: {len(hiragana_counts)}")
    print(f"  Samples per char: {np.mean(counts_h):.1f} ± {np.std(counts_h):.1f}")
    print(f"  Min: {min(counts_h)}, Max: {max(counts_h)}")
    
    print(f"\nKatakana Statistics:")
    print(f"  Characters: {len(katakana_counts)}")
    print(f"  Samples per char: {np.mean(counts_k):.1f} ± {np.std(counts_k):.1f}")
    print(f"  Min: {min(counts_k)}, Max: {max(counts_k)}")


if __name__ == "__main__":
    import os
    
    # Check if dataset exists
    dataset_path = 'data/processed/kana_dataset.npz'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run 'python data/extract_data.py' first to create the dataset.")
        exit(1)
    
    print("="*60)
    print("DATASET VISUALIZATION")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # 1. Show random samples from dataset
    print("1. Generating random sample visualization...")
    visualize_random_samples(dataset_path, num_samples=20)
    print()
    
    # 2. Show augmentation variations for a specific character
    print("2. Generating augmentation comparison...")
    visualize_augmentation_comparison(dataset_path, char_code=1, num_variations=10)
    print()
    
    # 3. Show class distribution
    print("3. Generating class distribution...")
    visualize_class_distribution(dataset_path)
    print()
    
    print("="*60)
    print("All visualizations completed!")
    print("Check data/processed/ folder for generated images.")
    print("="*60)