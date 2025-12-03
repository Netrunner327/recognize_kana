import numpy as np
from sklearn.model_selection import train_test_split

def create_train_val_test_split(dataset_path='data/processed/kana_dataset.npz',
                                train_ratio=0.7,
                                val_ratio=0.15,
                                test_ratio=0.15,
                                random_state=42):
    """
    Split dataset into train/validation/test sets
    
    Args:
        dataset_path: Path to kana_dataset.npz
        train_ratio: Proportion for training (0.7 = 70%)
        val_ratio: Proportion for validation (0.15 = 15%)
        test_ratio: Proportion for testing (0.15 = 15%)
        random_state: Random seed for reproducibility
    """
    print("Loading dataset...")
    data = np.load(dataset_path)
    
    # Load hiragana and katakana
    hiragana_images = data['hiragana_images']
    hiragana_codes = data['hiragana_codes']
    katakana_images = data['katakana_images']
    katakana_codes = data['katakana_codes']
    
    print(f"Loaded {len(hiragana_images):,} hiragana + {len(katakana_images):,} katakana")
    
    # Combine hiragana and katakana
    # Hiragana: codes 0-45, Katakana: codes 46-91
    all_images = np.concatenate([hiragana_images, katakana_images])
    all_labels = np.concatenate([
        hiragana_codes,  # 0-45
        katakana_codes + 46  # 46-91
    ])
    
    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(all_images):,}")
    print(f"  Classes: 92 (0-45 hiragana, 46-91 katakana)")
    print(f"  Image shape: {all_images.shape[1:]}")
    
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_images, all_labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=all_labels  # Ensure balanced split across all 92 classes
    )
    
    # Second split: separate validation from train
    # val_ratio / (train_ratio + val_ratio) to get correct proportion
    val_size = val_ratio / (train_ratio + val_ratio)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train:      {len(X_train):,} samples ({len(X_train)/len(all_images)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(all_images)*100:.1f}%)")
    print(f"  Test:       {len(X_test):,} samples ({len(X_test)/len(all_images)*100:.1f}%)")
    
    # Verify class distribution
    from collections import Counter
    train_dist = Counter(y_train)
    val_dist = Counter(y_val)
    test_dist = Counter(y_test)
    
    print(f"\nClass distribution check:")
    print(f"  Train: {len(train_dist)} classes, {min(train_dist.values())}-{max(train_dist.values())} samples/class")
    print(f"  Val:   {len(val_dist)} classes, {min(val_dist.values())}-{max(val_dist.values())} samples/class")
    print(f"  Test:  {len(test_dist)} classes, {min(test_dist.values())}-{max(test_dist.values())} samples/class")
    
    # Save splits
    output_path = 'data/processed/kana_train_val_test.npz'
    print(f"\nSaving splits to {output_path}...")
    
    np.savez_compressed(output_path,
                       X_train=X_train, y_train=y_train,
                       X_val=X_val, y_val=y_val,
                       X_test=X_test, y_test=y_test)
    
    print("✓ Splits saved successfully!")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


if __name__ == "__main__":
    create_train_val_test_split(
        dataset_path='data/processed/kana_dataset.npz',
        train_ratio=0.7,   # 70% for training
        val_ratio=0.15,    # 15% for validation (hyperparameter tuning)
        test_ratio=0.15,   # 15% for final test (never seen during training)
        random_state=42
    )