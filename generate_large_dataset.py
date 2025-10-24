import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_diverse_data(n_samples=1000000):
    """
    Sinh dữ liệu đa dạng với 5 features và 3 classes (0, 1, 2)
    Mỗi class có các patterns và phân phối khác nhau
    """
    print(f"Đang sinh {n_samples:,} records dữ liệu...")
    
    # Số lượng mẫu cho mỗi class (phân phối không đồng đều để đa dạng hơn)
    n_class0 = int(n_samples * 0.35)  # 35%
    n_class1 = int(n_samples * 0.40)  # 40%
    n_class2 = n_samples - n_class0 - n_class1  # 25%
    
    data_list = []
    
    # Class 0: Features có xu hướng âm và dương xen kẽ
    print("Sinh dữ liệu cho Class 0...")
    for _ in tqdm(range(n_class0), desc="Class 0"):
        feature1 = np.random.normal(-0.5, 1.2)
        feature2 = np.random.normal(2.0, 1.5)
        feature3 = np.random.uniform(-1.0, 3.5)
        feature4 = np.random.normal(-0.8, 1.0)
        feature5 = np.random.exponential(1.5) * np.random.choice([1, -1])
        data_list.append([feature1, feature2, feature3, feature4, feature5, 0])
    
    # Class 1: Features có xu hướng dương nhẹ
    print("Sinh dữ liệu cho Class 1...")
    for _ in tqdm(range(n_class1), desc="Class 1"):
        feature1 = np.random.normal(0.8, 1.0)
        feature2 = np.random.normal(0.5, 1.3)
        feature3 = np.random.normal(-1.2, 1.5)
        feature4 = np.random.uniform(0.0, 2.5)
        feature5 = np.random.gamma(2, 0.8) * np.random.choice([1, -1])
        data_list.append([feature1, feature2, feature3, feature4, feature5, 1])
    
    # Class 2: Features có giá trị lớn và phân tán
    print("Sinh dữ liệu cho Class 2...")
    for _ in tqdm(range(n_class2), desc="Class 2"):
        feature1 = np.random.normal(1.5, 0.8)
        feature2 = np.random.normal(-0.9, 1.2)
        feature3 = np.random.normal(2.5, 1.3)
        feature4 = np.random.normal(-0.5, 1.5)
        feature5 = np.random.normal(-1.0, 1.2)
        data_list.append([feature1, feature2, feature3, feature4, feature5, 2])
    
    # Trộn dữ liệu để không bị sắp xếp theo class
    print("Đang trộn dữ liệu...")
    np.random.shuffle(data_list)
    
    # Tạo DataFrame
    print("Đang tạo DataFrame...")
    df = pd.DataFrame(data_list, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'target'])
    
    # Làm tròn đến 1 chữ số thập phân để file nhỏ gọn hơn
    for col in ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']:
        df[col] = df[col].round(1)
    
    return df

def main():
    print("=" * 60)
    print("SINH DỮ LIỆU ĐA DẠNG CHO CNN PREDICTION")
    print("=" * 60)
    
    # Sinh 1 triệu records
    df = generate_diverse_data(n_samples=1000000)
    
    # Thống kê
    print("\n" + "=" * 60)
    print("THỐNG KÊ DỮ LIỆU")
    print("=" * 60)
    print(f"Tổng số records: {len(df):,}")
    print(f"\nPhân phối classes:")
    print(df['target'].value_counts().sort_index())
    
    print(f"\nThống kê các features:")
    print(df.describe())
    
    # Lưu file
    output_file = 'data_predict_dnn.csv'
    print(f"\nĐang lưu vào file {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Hoàn thành! Đã tạo file {output_file} với {len(df):,} records")
    print(f"Kích thước file: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB (trong bộ nhớ)")

if __name__ == "__main__":
    main()

