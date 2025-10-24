import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def build_neural_network(input_dim, num_classes):
    """
    Xây dựng mạng neural network cho dữ liệu tabular
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def load_and_preprocess_data(csv_file):
    """
    Đọc và xử lý dữ liệu từ file CSV
    """
    print(f"Đang đọc dữ liệu từ {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Đã tải {len(df):,} records")
    print(f"\nThống kê dữ liệu:")
    print(df.describe())
    print(f"\nPhân phối classes:")
    print(df['target'].value_counts().sort_index())
    
    # Tách features và target
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Chia train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nKích thước train set: {X_train.shape}")
    print(f"Kích thước test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def visualize_training_history(history, save_dir='visualizations'):
    """
    Visualize training history (loss và accuracy)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#2E86AB')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#A23B72')
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ training history: {save_path}")
    plt.close()

def visualize_confusion_matrix(y_true, y_pred, class_names=None, save_dir='visualizations'):
    """
    Visualize confusion matrix
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)),
                linewidths=0.5, linecolor='gray')
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    # Thêm tỷ lệ accuracy cho mỗi class
    for i in range(len(cm)):
        accuracy = cm[i, i] / cm[i, :].sum() * 100
        plt.text(len(cm) + 0.5, i + 0.5, f'{accuracy:.1f}%', 
                va='center', ha='left', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu confusion matrix: {save_path}")
    plt.close()

def visualize_class_distribution(y_train, y_test, save_dir='visualizations'):
    """
    Visualize phân phối classes trong train và test set
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train set distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    colors_train = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = axes[0].bar(unique_train, counts_train, color=colors_train[:len(unique_train)], 
                        edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[0].set_title('Class Distribution - Training Set', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Thêm labels và phần trăm
    for i, (bar, count) in enumerate(zip(bars1, counts_train)):
        height = bar.get_height()
        percentage = (count / counts_train.sum()) * 100
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Test set distribution
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    bars2 = axes[1].bar(unique_test, counts_test, color=colors_train[:len(unique_test)], 
                        edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1].set_title('Class Distribution - Test Set', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Thêm labels và phần trăm
    for i, (bar, count) in enumerate(zip(bars2, counts_test)):
        height = bar.get_height()
        percentage = (count / counts_test.sum()) * 100
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ phân phối classes: {save_path}")
    plt.close()

def visualize_predictions_distribution(y_true, y_pred, save_dir='visualizations'):
    """
    Visualize phân phối predictions so với actual labels
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted counts
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    actual_counts = [np.sum(y_true == c) for c in unique_classes]
    pred_counts = [np.sum(y_pred == c) for c in unique_classes]
    
    x = np.arange(len(unique_classes))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, actual_counts, width, label='Actual', 
                       color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = axes[0].bar(x + width/2, pred_counts, width, label='Predicted', 
                       color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    axes[0].set_title('Actual vs Predicted Counts', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(unique_classes)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Thêm values lên bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9)
    
    # Prediction accuracy per class
    accuracies = []
    for c in unique_classes:
        mask = y_true == c
        if mask.sum() > 0:
            acc = np.sum((y_true[mask] == y_pred[mask])) / mask.sum() * 100
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    colors_acc = ['#2ecc71' if acc >= 80 else '#f39c12' if acc >= 60 else '#e74c3c' 
                  for acc in accuracies]
    bars3 = axes[1].bar(unique_classes, accuracies, color=colors_acc, 
                        edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1].set_title('Accuracy Per Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_ylim([0, 105])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Thêm percentage labels
    for bar, acc in zip(bars3, accuracies):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'predictions_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ phân tích predictions: {save_path}")
    plt.close()

def create_summary_report(history, y_true, y_pred, num_classes, save_dir='visualizations'):
    """
    Tạo báo cáo tổng hợp với classification report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    
    # Title
    fig.suptitle('Model Performance Summary Report', fontsize=18, fontweight='bold', y=0.98)
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Tạo table data - lọc chỉ lấy các class numbers
    classes = []
    for k in report.keys():
        if k not in ['accuracy', 'macro avg', 'weighted avg']:
            try:
                # Thử chuyển sang int để đảm bảo đây là class number
                int(k)
                classes.append(k)
            except ValueError:
                pass
    classes = sorted(classes, key=lambda x: int(x))
    
    table_data = []
    
    for cls in classes:
        if cls in report:
            table_data.append([
                f'Class {cls}',
                f"{report[cls]['precision']:.3f}",
                f"{report[cls]['recall']:.3f}",
                f"{report[cls]['f1-score']:.3f}",
                f"{int(report[cls]['support'])}"
            ])
    
    # Thêm averages
    table_data.append(['', '', '', '', ''])
    if 'macro avg' in report:
        table_data.append([
            'Macro Avg',
            f"{report['macro avg']['precision']:.3f}",
            f"{report['macro avg']['recall']:.3f}",
            f"{report['macro avg']['f1-score']:.3f}",
            ''
        ])
    if 'weighted avg' in report:
        table_data.append([
            'Weighted Avg',
            f"{report['weighted avg']['precision']:.3f}",
            f"{report['weighted avg']['recall']:.3f}",
            f"{report['weighted avg']['f1-score']:.3f}",
            f"{int(report['weighted avg']['support'])}"
        ])
    
    # Overall accuracy
    overall_acc = np.sum(y_true == y_pred) / len(y_true) * 100
    
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Thông tin chung
    info_text = f"""
    📊 TRAINING SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Number of Classes: {num_classes}
    • Total Test Samples: {len(y_true):,}
    • Overall Accuracy: {overall_acc:.2f}%
    • Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%
    • Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%
    • Final Training Loss: {history.history['loss'][-1]:.4f}
    • Final Validation Loss: {history.history['val_loss'][-1]:.4f}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📈 DETAILED METRICS (Classification Report)
    """
    
    ax.text(0.1, 0.95, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')
    
    # Table
    table = ax.table(cellText=table_data,
                    colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0.1, 0.1, 0.8, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            if i > len(classes):
                table[(i, j)].set_facecolor('#95a5a6')
                table[(i, j)].set_text_props(weight='bold')
    
    save_path = os.path.join(save_dir, 'summary_report.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu báo cáo tổng hợp: {save_path}")
    plt.close()

def main():
    # Đọc dữ liệu từ CSV
    csv_file = 'data_predict_dnn.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file)
    
    # Thông số model
    input_dim = X_train.shape[1]  # Số lượng features
    num_classes = len(np.unique(y_train))  # Số lượng classes
    
    print(f"\n{'='*60}")
    print(f"XÂY DỰNG MODEL")
    print(f"{'='*60}")
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Xây dựng model
    model = build_neural_network(input_dim, num_classes)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n" + "="*60)
    print("KIẾN TRÚC MODEL")
    print("="*60)
    model.summary()

    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    print("\n" + "="*60)
    print("ĐÁNH GIÁ MODEL")
    print("="*60)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    print("\nĐang tạo predictions cho test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report text
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    # Visualizations
    print("\n" + "="*60)
    print("TẠO VISUALIZATIONS")
    print("="*60)
    
    save_dir = 'visualizations'
    
    # 1. Training history
    visualize_training_history(history, save_dir)
    
    # 2. Class distribution
    visualize_class_distribution(y_train, y_test, save_dir)
    
    # 3. Confusion matrix
    class_names = [f'Class {i}' for i in range(num_classes)]
    visualize_confusion_matrix(y_test, y_pred, class_names, save_dir)
    
    # 4. Predictions analysis
    visualize_predictions_distribution(y_test, y_pred, save_dir)
    
    # 5. Summary report
    create_summary_report(history, y_test, y_pred, num_classes, save_dir)
    
    print("\n" + "="*60)
    print("✓ Đã tạo tất cả visualizations!")
    print(f"✓ Các hình ảnh được lưu trong thư mục: {save_dir}/")
    print("="*60)
    
    # Lưu model
    model_path = 'dnn_predict_model.h5'
    model.save(model_path)
    print(f"\n✓ Đã lưu model vào file: {model_path}")
    
    # Tổng kết
    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH!")
    print("="*60)
    print(f"📊 Test Accuracy: {test_acc*100:.2f}%")
    print(f"📈 Các visualizations: {save_dir}/")
    print(f"💾 Model được lưu tại: {model_path}")
    print("="*60)

if __name__ == "__main__":
    main()
