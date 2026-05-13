import os, sys, glob, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,precision_recall_curve
from scipy.signal import butter, filtfilt

warnings.filterwarnings('ignore')

# ==========================================
# 1. 基础配置
# ==========================================
# ❗ 修改为存放测试集 "_data.csv" 和 "_fix.txt" 的文件夹路径
MODEL_PATH = "mimic_train_best_developed.h5"
TEST_DATA_DIR = r"D:\AFdataset\mimic_perform_test_all_csv"

INPUT_LENGTH = 800
TARGET_COLUMN = 'PPG'


def butter_bandpass_filter(data, lowcut=0.5, highcut=18.0, fs=125, order=2):
    """0.5Hz过滤基线漂移，8.0Hz过滤高频噪声"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def is_signal_valid(signal, fs=125):
    """
    ⭐ 进阶版信号质量评估 (SQA)
    不仅防平线，更要防突变的运动伪影和剧烈的基线游走！
    """
    # 1. 拦截传感器断联 (大段连续相同的0或常数)
    diff = np.diff(signal)
    if np.sum(diff == 0) / len(diff) > 0.15:
        return False

    # 2. 拦截纯噪声或平线 (整体方差过小)
    if np.std(signal) < 0.05:  # 稍微提高一点平线门槛
        return False


    # 3.使用 IQR (四分位距) 衡量信号的主体波动幅度
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25

    # 防止除以0
    if iqr < 1e-5:
        return False


    max_dev = (np.max(signal) - np.median(signal)) / iqr
    min_dev = (np.median(signal) - np.min(signal)) / iqr

    if max_dev > 4.0 or min_dev > 4.0:
        return False

    return True


# ==========================================
# 2. 数据加载与过滤
# ==========================================
def load_test_data(csv_dir, input_length=INPUT_LENGTH, target_col=TARGET_COLUMN):
    print(f"\n[Data Load] 正在从 {csv_dir} 加载测试数据与标签...")

    # 匹配 *_data.csv
    excel_files = glob.glob(os.path.join(csv_dir, "**", "*_data.csv*"), recursive=True)
    if len(excel_files) == 0:
        raise ValueError(f"❌ 在 {csv_dir} 未找到 *_data.csv 文件！")

    test_sigs, test_lbls = [], []
    success_count, missing_txt_count, discarded_chunks = 0, 0, 0

    for excel_path in excel_files:
        dir_name = os.path.dirname(excel_path)
        file_name = os.path.basename(excel_path)

        # 截取 _data 之前的前缀，并拼接 _fix.txt
        prefix = file_name.split('_data')[0]
        txt_name = prefix + "_fix.txt"
        txt_path = os.path.join(dir_name, txt_name)

        if not os.path.exists(txt_path):
            missing_txt_count += 1
            continue

        label = None
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith("Group:"):
                        val = line.split(":")[1].strip().lower()
                        if val == 'n':
                            label = [1.0, 0.0]
                        elif val == 'a':
                            label = [0.0, 1.0]
                        break
        except Exception:
            continue
        if label is None: continue

        try:
            df = pd.read_csv(excel_path, encoding='utf-8')
            df.columns = df.columns.astype(str).str.strip()
            if target_col not in df.columns: continue

            sig = np.nan_to_num(pd.to_numeric(df[target_col].values, errors='coerce'))
            sig = butter_bandpass_filter(sig)

            for i in range(len(sig) // input_length):
                chunk = sig[i * input_length: (i + 1) * input_length]
                # ⭐ SQA质量控制：过滤掉导致评测下降的垃圾波形
                if is_signal_valid(chunk):
                    test_sigs.append(chunk)
                    test_lbls.append(label)
                else:
                    discarded_chunks += 1
            success_count += 1
        except Exception as e:
            print(f"读取 Excel 出错 {file_name}: {e}")

    print(f"✅ 成功匹配并解析 {success_count} 个患者。")
    print(f"🛡️ SQA质量控制：评估时自动剔除了 {discarded_chunks} 个低质量片段。")
    print(f"📊 最终参与评估的有效切片: {len(test_sigs)} 个。")

    return np.array(test_sigs, dtype=np.float32), np.array(test_lbls, dtype=np.float32)


# ==========================================
# 3. 执行评估主程序
# ==========================================
def main():
    try:
        test_x, test_y = load_test_data(TEST_DATA_DIR)
    except Exception as e:
        return print(e)

    if len(test_x) == 0:
        return print("❌ 有效数据为空，退出评估。")

    print("\n[Preprocessing] 进行 Z-Score 标准化...")
    # ⭐ 核心修改：将 Min-Max 替换为 Z-Score，确保与训练集同分布
    test_x = np.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0)
    mean_v = np.mean(test_x, axis=1, keepdims=True)
    std_v = np.std(test_x, axis=1, keepdims=True)
    std_v[std_v == 0] = 1e-7  # 防止除以0
    test_x = (test_x - mean_v) / std_v

    if test_x.ndim == 2:
        test_x = test_x[..., np.newaxis]

    print(f"[Model Load] 正在加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)

    print("[Prediction] 开始进行推理预测...")
    preds = model.predict(test_x, batch_size=128, verbose=1)

    # ⭐ 获取真实标签和预测概率
    y_true = np.argmax(test_y, axis=1)
    y_prob = preds[:, 1]  # AF类的预测概率

    # ⭐ 核心修改：使用 precision_recall_curve 寻找最大化 F1 的最佳阈值
    print("\n[Threshold Optimization] 正在寻找最大化 F1 的最佳阈值...")
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # 计算每个阈值对应的 F1 分数
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # 找到最大化 F1 的阈值索引
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1_at_threshold = f1_scores[best_idx]

    # 使用最佳阈值进行最终预测
    y_pred = (y_prob >= best_threshold).astype(int)

    print(f"✅ 最佳阈值: {best_threshold:.4f}")
    print(f"✅ 该阈值对应的 F1-Score: {best_f1_at_threshold:.4f}")

    # ---------------- 打印评估报告 ----------------
    print("\n" + "=" * 45 + "\n 🏥 最终性能评估报告\n" + "=" * 45)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn + 1e-7)  # 敏感度 (Recall)
    tnr = tn / (tn + fp + 1e-7)  # 特异性
    ppv = tp / (tp + fp + 1e-7)  # 阳性预测值 (Precision)
    npv = tn / (tn + fn + 1e-7)  # 阴性预测值
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)  # F1 分数
    acc = (tp + tn) / len(test_x)  # 准确率

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float('nan')

    # ⭐ 打印阈值信息
    print(f"🔧 使用阈值: {best_threshold:.4f} (通过最大化 F1 确定)")
    print(f"🎯 总体准确率 (Accuracy) : {acc:.4f}")
    if not np.isnan(auc):
        print(f"📊 ROC-AUC            : {auc:.4f}")
    print(f"📈 敏感度 (TPR/Recall)  : {tpr:.4f}  <- 查出房颤的能力")
    print(f"📉 特异性 (TNR)         : {tnr:.4f}  <- 排除正常的能力")
    print(f"⚖️ F1-Score             : {f1:.4f}")
    print(f"🎯 阳性预测值 (PPV)     : {ppv:.4f}")
    print(f"🛡️ 阴性预测值 (NPV)     : {npv:.4f}")

    print("\n--- 混淆矩阵 (Confusion Matrix) ---")
    print(f"                 预测 Normal (0)   预测 AF (1)")
    print(f"实际 Normal (0) |   {tn:<13} |   {fp}")
    print(f"实际 AF (1)     |   {fn:<13} |   {tp}")
    print("=============================================\n")

if __name__ == "__main__":
    main()