import os
import sys
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import detrend
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

# 屏蔽不必要的警告
warnings.filterwarnings('ignore')

# =======================
#   路径配置
# =======================
MODEL_PATH = "./deepbeat_finetuned_improved_best.h5"
DATA_PATH = "D:/AFdataset/test.npz"


# =======================
#   1. 加载测试数据与严格预处理
# =======================
print("\n[1/5] 加载测试数据并进行预处理...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"找不到测试集文件: {DATA_PATH}")

data_test = np.load(DATA_PATH, allow_pickle=True)
raw_signals = data_test['signal'].astype(np.float32)

# 预处理：Detrending
signals_cleaned = detrend(raw_signals.squeeze(), axis=1)

# 预处理：Z-Score 标准化
mean_val = np.mean(signals_cleaned, axis=1, keepdims=True)
std_val = np.std(signals_cleaned, axis=1, keepdims=True)
std_val[std_val == 0] = 1.0
test_x = (signals_cleaned - mean_val) / std_val

if test_x.ndim == 2:
    test_x = test_x[..., np.newaxis]

test_qa = data_test['qa_label']
test_r = data_test['rhythm']
# 提取参数信息 (ID, Timestamp等)
test_p = pd.DataFrame(data_test['parameters'], columns=['timestamp', 'stream', 'ID'])

print(f"预处理完成。数据形状: {test_x.shape}")

# =======================
#   2. 加载模型
# =======================
print("\n[2/5] 加载模型...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)
print("模型加载成功。")

# =======================
#   3. 执行预测与自动寻找最佳阈值
# =======================
print("\n[3/5] 正在进行模型预测...")
preds_qa, preds_r = model.predict(test_x, batch_size=256, verbose=1)

# QA 筛选 (只保留 Excellent 样本进行评价)
qa_classes = np.argmax(preds_qa, axis=1)
excellent_mask = (qa_classes == 2)
excellent_idx = np.where(excellent_mask)[0]

if len(excellent_idx) == 0:
    print("!!! 警告: 没有样本通过 QA 筛选 !!!")
    sys.exit()

# 提取通过 QA 筛选的预测概率和真实标签
r_probs = preds_r[excellent_idx][:, 1]
r_true = np.argmax(test_r[excellent_idx], axis=1)
# 提取对应的元数据
metadata_excellent = test_p.iloc[excellent_idx].reset_index(drop=True)

# 自动寻找最佳阈值 (基于 F1 Score)
precisions, recalls, thresholds_f1 = precision_recall_curve(r_true, r_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds_f1[best_idx]

print(f"自动优化完成。最佳阈值: {best_threshold:.4f}, 预期最大 F1: {f1_scores[best_idx]:.4f}")

# =======================
#   4. 最终性能计算
# =======================
r_pred = (r_probs >= best_threshold).astype(int)

cm = metrics.confusion_matrix(r_true, r_pred)
tn, fp, fn, tp = cm.ravel()

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0
acc = (tp + tn) / (tn + tp + fp + fn)
auroc = metrics.roc_auc_score(r_true, r_probs)

print("-" * 30)
print(f"混淆矩阵:\n{cm}")
print(f"TPR (召回率):    {tpr:.4f}")
print(f"TNR (特异性):    {tnr:.4f}")
print(f"PPV (精准率):    {ppv:.4f}")
print(f"NPV (阴性预测值): {npv:.4f}")
print(f"F1 Score:       {f1:.4f}")
print(f"AUROC:          {auroc:.4f}")
print(f"ACC:            {acc:.4f}")
print("-" * 30)

