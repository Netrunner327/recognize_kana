import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift, zoom
import random

def augment_image(img_array, num_variations=5):
    """Create augmented versions with visualizations"""
    augmented = []
    
    for i in range(num_variations):
        img = img_array.copy()
        
        # Random rotation
        angle = random.uniform(-15, 15)
        img = rotate(img, angle, reshape=False, cval=0)
        
        # Random shift
        shift_x = random.randint(-3, 3)
        shift_y = random.randint(-3, 3)
        img = shift(img, [shift_y, shift_x], cval=0)
        
        # Random zoom
        zoom_factor = random.uniform(0.9, 1.1)
        img = zoom(img, zoom_factor, order=1)
        
        # Crop/pad back to 64x64
        if img.shape[0] > 64:
            start = (img.shape[0] - 64) // 2
            img = img[start:start+64, start:start+64]
        elif img.shape[0] < 64:
            pad_width = ((64 - img.shape[0]) // 2, (64 - img.shape[0] + 1) // 2)
            img = np.pad(img, (pad_width, pad_width), mode='constant')[:64, :64]
        
        # Random noise
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img + noise, 0, 255)
        
        augmented.append(img)
    
    return augmented

# Load one sample from your dataset
data = np.load('data/processed/kana_dataset.npz')
original = data['katakana_images'][0] * 255  # Denormalize for visualization

# Create augmented versions
augmented_samples = augment_image(original, num_variations=8)

# Visualize
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Augmentation Examples: Original vs Variations', fontsize=16)

# Original
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('ORIGINAL', fontweight='bold')
axes[0, 0].axis('off')

# Augmented versions
for idx, aug_img in enumerate(augmented_samples):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(aug_img, cmap='gray')
    axes[row, col].set_title(f'Variation {idx + 1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('data/processed/augmentation_examples.png', dpi=150)
print("Visualization saved to data/processed/augmentation_examples.png")
plt.show()