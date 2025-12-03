import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Romanized kana labels
HIRAGANA_ROMANJI = [
    'a', 'i', 'u', 'e', 'o',
    'ka', 'ki', 'ku', 'ke', 'ko',
    'sa', 'shi', 'su', 'se', 'so',
    'ta', 'chi', 'tsu', 'te', 'to',
    'na', 'ni', 'nu', 'ne', 'no',
    'ha', 'hi', 'fu', 'he', 'ho',
    'ma', 'mi', 'mu', 'me', 'mo',
    'ya', 'yu', 'yo',
    'ra', 'ri', 'ru', 're', 'ro',
    'wa', 'wo', 'n'
]

KATAKANA_ROMANJI = [
    'A', 'I', 'U', 'E', 'O',
    'KA', 'KI', 'KU', 'KE', 'KO',
    'SA', 'SHI', 'SU', 'SE', 'SO',
    'TA', 'CHI', 'TSU', 'TE', 'TO',
    'NA', 'NI', 'NU', 'NE', 'NO',
    'HA', 'HI', 'FU', 'HE', 'HO',
    'MA', 'MI', 'MU', 'ME', 'MO',
    'YA', 'YU', 'YO',
    'RA', 'RI', 'RU', 'RE', 'RO',
    'WA', 'WO', 'N'
]

ALL_KANA_LABELS = HIRAGANA_ROMANJI + KATAKANA_ROMANJI


def evaluate_model(model_path='models/kana_best.h5', dataset_path='../data/processed/kana_train_val_test.npz'):
    """
    Comprehensive model evaluation with confusion matrix and per-class accuracy
    
    Args:
        model_path: Path to trained model
        dataset_path: Path to test dataset
    """
    print("="*60)
    print("KANA RECOGNITION CNN - EVALUATION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Load test data
    print(f"\nLoading test data from {dataset_path}...")
    data = np.load(dataset_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Reshape for CNN
    X_test = X_test.reshape(-1, 64, 64, 1)
    
    print(f"✓ Test set: {len(X_test):,} samples, 92 classes")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n{'='*60}")
    print(f"OVERALL TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    print("-" * 60)
    
    hiragana_correct = 0
    hiragana_total = 0
    katakana_correct = 0
    katakana_total = 0
    
    per_class_acc = []
    worst_classes = []
    
    for class_id in range(92):
        mask = y_test == class_id
        if mask.sum() == 0:
            continue
            
        class_correct = (y_pred[mask] == y_test[mask]).sum()
        class_total = mask.sum()
        class_acc = class_correct / class_total
        
        per_class_acc.append((class_id, class_acc, class_total))
        
        # Track hiragana vs katakana
        if class_id < 46:  # Hiragana
            hiragana_correct += class_correct
            hiragana_total += class_total
        else:  # Katakana
            katakana_correct += class_correct
            katakana_total += class_total
        
        # Track worst performing classes
        if class_acc < 0.95:
            worst_classes.append((class_id, class_acc, class_total))
    
    # Print hiragana vs katakana performance
    print(f"\nHiragana accuracy: {hiragana_correct/hiragana_total*100:.2f}% ({hiragana_correct}/{hiragana_total})")
    print(f"Katakana accuracy: {katakana_correct/katakana_total*100:.2f}% ({katakana_correct}/{katakana_total})")
    
    # Print worst performing classes
    if worst_classes:
        print(f"\nClasses with < 95% accuracy:")
        worst_classes.sort(key=lambda x: x[1])
        for class_id, acc, total in worst_classes[:10]:
            label = ALL_KANA_LABELS[class_id]
            print(f"  {label:>5} (class {class_id:2d}): {acc*100:.2f}% ({int(acc*total)}/{total})")
    else:
        print(f"\n✓ All classes have ≥95% accuracy!")
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, ALL_KANA_LABELS)
    
    # Generate classification report
    print("\nClassification Report (top 10 and bottom 10 classes):")
    print("-" * 80)
    report = classification_report(y_test, y_pred, target_names=ALL_KANA_LABELS, 
                                   output_dict=True, zero_division=0)
    
    # Sort by f1-score
    class_scores = [(name, metrics['f1-score'], metrics['support']) 
                    for name, metrics in report.items() 
                    if name in ALL_KANA_LABELS]
    class_scores.sort(key=lambda x: x[1])
    
    print("\nWorst 10 classes (by F1-score):")
    for name, f1, support in class_scores[:10]:
        print(f"  {name:>5}: F1={f1:.4f}, Support={int(support)}")
    
    print("\nBest 10 classes (by F1-score):")
    for name, f1, support in class_scores[-10:]:
        print(f"  {name:>5}: F1={f1:.4f}, Support={int(support)}")
    
    # Show misclassified examples
    print("\nGenerating misclassification examples...")
    plot_misclassified_examples(X_test, y_test, y_pred, y_pred_probs, ALL_KANA_LABELS)
    
    # Prediction confidence analysis
    print("\nAnalyzing prediction confidence...")
    analyze_confidence(y_test, y_pred, y_pred_probs)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. evaluation/confusion_matrix.png - Full confusion matrix")
    print("  2. evaluation/confusion_matrix_hiragana.png - Hiragana only")
    print("  3. evaluation/confusion_matrix_katakana.png - Katakana only")
    print("  4. evaluation/misclassified_examples.png - Examples of errors")
    print("  5. evaluation/confidence_distribution.png - Prediction confidence")


def plot_confusion_matrix(y_true, y_pred, labels, save_dir='evaluation'):
    """
    Plot confusion matrix (full, hiragana-only, katakana-only)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Full confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - All 92 Kana', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved {save_dir}/confusion_matrix.png")
    plt.close()
    
    # Hiragana only
    hiragana_mask = y_true < 46
    if hiragana_mask.sum() > 0:
        cm_hira = confusion_matrix(y_true[hiragana_mask], y_pred[hiragana_mask], 
                                    labels=list(range(46)))
        
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm_hira, annot=False, fmt='d', cmap='Blues',
                    xticklabels=HIRAGANA_ROMANJI, yticklabels=HIRAGANA_ROMANJI,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Hiragana Only', fontsize=16, pad=20)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix_hiragana.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_dir}/confusion_matrix_hiragana.png")
        plt.close()
    
    # Katakana only
    katakana_mask = y_true >= 46
    if katakana_mask.sum() > 0:
        cm_kata = confusion_matrix(y_true[katakana_mask] - 46, y_pred[katakana_mask] - 46,
                                    labels=list(range(46)))
        
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm_kata, annot=False, fmt='d', cmap='Reds',
                    xticklabels=KATAKANA_ROMANJI, yticklabels=KATAKANA_ROMANJI,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Katakana Only', fontsize=16, pad=20)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix_katakana.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_dir}/confusion_matrix_katakana.png")
        plt.close()


def plot_misclassified_examples(X_test, y_true, y_pred, y_pred_probs, labels, 
                                n_examples=20, save_dir='evaluation'):
    """
    Show examples of misclassified characters
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("✓ No misclassifications found!")
        return
    
    # Select random misclassified samples
    n_examples = min(n_examples, len(misclassified_idx))
    selected_idx = np.random.choice(misclassified_idx, n_examples, replace=False)
    
    # Plot
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(f'Misclassified Examples ({len(misclassified_idx)} total errors)', 
                 fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(selected_idx):
        row = i // 5
        col = i % 5
        
        img = X_test[idx].squeeze()
        true_label = labels[y_true[idx]]
        pred_label = labels[y_pred[idx]]
        confidence = y_pred_probs[idx, y_pred[idx]] * 100
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                                 fontsize=9, color='red')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/misclassified_examples.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved {save_dir}/misclassified_examples.png")
    plt.close()


def analyze_confidence(y_true, y_pred, y_pred_probs, save_dir='evaluation'):
    """
    Analyze prediction confidence distribution
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get confidence for each prediction
    confidences = np.max(y_pred_probs, axis=1)
    correct_mask = y_true == y_pred
    
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(correct_confidences, bins=50, alpha=0.7, label='Correct', color='green')
    ax1.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', color='red')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot([correct_confidences, incorrect_confidences], 
                labels=['Correct', 'Incorrect'])
    ax2.set_ylabel('Confidence')
    ax2.set_title('Confidence: Correct vs Incorrect Predictions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confidence_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved {save_dir}/confidence_distribution.png")
    plt.close()
    
    print(f"\nConfidence statistics:")
    print(f"  Correct predictions:   {correct_confidences.mean()*100:.2f}% ± {correct_confidences.std()*100:.2f}%")
    print(f"  Incorrect predictions: {incorrect_confidences.mean()*100:.2f}% ± {incorrect_confidences.std()*100:.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Kana Recognition CNN')
    parser.add_argument('--model', type=str, default='models/kana_best.h5', 
                       help='Path to model file')
    parser.add_argument('--dataset', type=str, default='../data/processed/kana_train_val_test.npz',
                       help='Path to dataset file')
    
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model, dataset_path=args.dataset)