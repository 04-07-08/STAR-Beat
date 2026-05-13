import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import umap
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
import os
import glob
import warnings
from scipy.signal import butter, filtfilt, detrend

warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 1. 基础配置
# ==========================================
TEST_DATA_DIR = r"D:\AFdataset\mimic_perform_train_all_csv"
MODEL_PATH = "mimic_train_best_developed_1.h5"

INPUT_LENGTH = 800
TARGET_COLUMN = 'PPG'


CLASS_NORMAL = 0
CLASS_AFIB = 1


def butter_bandpass_filter(data, lowcut=0.5, highcut=18.0, fs=125, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def is_signal_valid(signal, fs=125):
    diff = np.diff(signal)
    if np.sum(diff == 0) / len(diff) > 0.15: return False
    if np.std(signal) < 0.05: return False
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-5: return False
    max_dev = (np.max(signal) - np.median(signal)) / iqr
    min_dev = (np.median(signal) - np.min(signal)) / iqr
    if max_dev > 2.5 or min_dev > 2.5: return False
    return True


# ==========================================
# 2. 数据加载与预处理
# ==========================================
def load_test_data_for_vis(csv_dir, input_length=INPUT_LENGTH, target_col=TARGET_COLUMN):
    print(f"\n[1] 正在从 {csv_dir} 加载可视化数据...")
    excel_files = glob.glob(os.path.join(csv_dir, "**", "*_data.csv*"), recursive=True)
    if len(excel_files) == 0:
        raise ValueError(f"❌ 在 {csv_dir} 未找到 *_data.csv 文件！")

    test_sigs, test_lbls = [], []
    discarded_chunks = 0

    for excel_path in excel_files:
        dir_name = os.path.dirname(excel_path)
        file_name = os.path.basename(excel_path)

        prefix = file_name.split('_data')[0]
        txt_name = prefix + "_fix.txt"
        txt_path = os.path.join(dir_name, txt_name)

        if not os.path.exists(txt_path): continue

        label = None
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith("Group:"):
                        val = line.split(":")[1].strip().lower()
                        # 对齐模型训练时的真实权重索引
                        if val == 'n':
                            label = [1.0, 0.0]  # 正常心律 -> 索引0
                        elif val == 'a':
                            label = [0.0, 1.0]  # 房颤 -> 索引1
                        break
        except Exception:
            continue
        if label is None: continue

        try:
            df = pd.read_csv(excel_path, encoding='utf-8')
            df.columns = df.columns.astype(str).str.strip()
            if target_col not in df.columns: continue

            sig = np.nan_to_num(pd.to_numeric(df[target_col].values, errors='coerce'))
            sig = butter_bandpass_filter(sig, lowcut=0.5, highcut=18.0)
            sig = detrend(sig)

            for i in range(len(sig) // input_length):
                chunk = sig[i * input_length: (i + 1) * input_length]
                if is_signal_valid(chunk):
                    test_sigs.append(chunk)
                    test_lbls.append(label)
                else:
                    discarded_chunks += 1
        except Exception as e:
            pass

    print(f"🛡️ SQA质量控制：测试集自动丢弃了 {discarded_chunks} 个低质量片段。")
    return np.array(test_sigs, dtype=np.float32), np.array(test_lbls, dtype=np.float32)


X_test_all, y_test_all = load_test_data_for_vis(TEST_DATA_DIR)
if len(X_test_all) == 0:
    raise ValueError("未提取到任何有效数据，请检查路径或调整SQA阈值。")

print(f"[2] 准备对 {len(X_test_all)} 个信号片段进行 Z-Score 标准化...")
X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)
mean_v = np.mean(X_test_all, axis=1, keepdims=True)
std_v = np.std(X_test_all, axis=1, keepdims=True)
std_v[std_v == 0] = 1e-7
X_test_all = (X_test_all - mean_v) / std_v
X_test_all = X_test_all[..., np.newaxis]

print(f"[3] 正在加载模型: {MODEL_PATH} ...")
model = load_model(MODEL_PATH, compile=False)


# ==========================================
# 3. UMAP 学习表征降维可视化
# ==========================================
def plot_umap_representations(model, X, y_true_ohe):
    print("\n[UMAP] 提取深层特征用于流形降维可视化...")
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    if len(dense_layers) >= 2:
        feature_tensor = dense_layers[-2].output
    else:
        feature_tensor = dense_layers[-1].input

    feature_extractor = Model(inputs=model.input, outputs=feature_tensor)
    features = feature_extractor.predict(X, verbose=1)

    y_true_classes = np.argmax(y_true_ohe, axis=1)
    preds = model.predict(X, verbose=0)
    preds_classes = np.argmax(preds, axis=1)
    acc = np.mean(preds_classes == y_true_classes)
    print(f"  -> 该批次数据预测准确率: {acc:.2%}")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(features)

    # ⭐ 修正：使用正确的类别索引
    idx_normal = (y_true_classes == CLASS_NORMAL)
    idx_afib = (y_true_classes == CLASS_AFIB)

    plt.figure(figsize=(9, 7))
    plt.scatter(embedding[idx_normal, 0], embedding[idx_normal, 1],
                c='mediumseagreen', s=20, alpha=0.6, edgecolors='none', label='Normal (Group: n)')
    plt.scatter(embedding[idx_afib, 0], embedding[idx_afib, 1],
                c='crimson', s=20, alpha=0.8, edgecolors='none', label='Atrial Fibrillation (Group: a)')

    plt.title(f"UMAP Deep Feature Space (Acc: {acc:.1%})", fontsize=14)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.xticks([]);
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('Figure_UMAP_MIMIC.png', dpi=300)
    print("✅ 已保存 -> Figure_UMAP_MIMIC.png")

    plt.show()
    return embedding


# ==========================================
# 4. UMAP 重叠区域边界样本分析
# ==========================================
def plot_overlapping_waveforms(X, y_true_ohe, embedding, num_samples=5):
    print("\n[Overlap Analysis] 正在寻找 UMAP 空间中重叠/边界区域的样本...")
    y_true_classes = np.argmax(y_true_ohe, axis=1)

    # ⭐ 修正：使用正确的类别索引
    idx_normal = np.where(y_true_classes == CLASS_NORMAL)[0]
    idx_afib = np.where(y_true_classes == CLASS_AFIB)[0]

    emb_normal = embedding[idx_normal]
    emb_afib = embedding[idx_afib]

    dist_matrix = cdist(emb_normal, emb_afib, metric='euclidean')

    min_dist_normal_to_afib = np.min(dist_matrix, axis=1)
    closest_normal_local_idx = np.argsort(min_dist_normal_to_afib)[:num_samples]
    closest_normal_global_idx = idx_normal[closest_normal_local_idx]

    min_dist_afib_to_normal = np.min(dist_matrix, axis=0)
    closest_afib_local_idx = np.argsort(min_dist_afib_to_normal)[:num_samples]
    closest_afib_global_idx = idx_afib[closest_afib_local_idx]

    fig, axes = plt.subplots(2, num_samples, figsize=(16, 6))
    fig.suptitle("Overlapping Samples in UMAP Space\n(Top: Normal mistaken as AFib, Bottom: AFib mistaken as Normal)",
                 fontsize=14)

    for i in range(num_samples):
        if i < len(closest_normal_global_idx):
            n_idx = closest_normal_global_idx[i]
            dist_val = min_dist_normal_to_afib[closest_normal_local_idx[i]]
            ax = axes[0, i]
            ax.plot(X[n_idx].flatten(), color='mediumseagreen')
            ax.set_title(f"Normal (idx:{n_idx})\nDist to AFib: {dist_val:.2f}", fontsize=10)
            ax.set_xticks([]);
            ax.set_yticks([])

        if i < len(closest_afib_global_idx):
            a_idx = closest_afib_global_idx[i]
            dist_val = min_dist_afib_to_normal[closest_afib_local_idx[i]]
            ax = axes[1, i]
            ax.plot(X[a_idx].flatten(), color='crimson')
            ax.set_title(f"AFib (idx:{a_idx})\nDist to Normal: {dist_val:.2f}", fontsize=10)
            ax.set_xticks([]);
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('UMAP_Overlapping_Waveforms.png', dpi=300)
    print("✅ 重叠波形分析图已保存 -> UMAP_Overlapping_Waveforms.png")
    plt.show()


# ==========================================
# 5. 高级可解释性分析 (CAM, IG, Occlusion)
# ==========================================
def plot_saliency_map(signal, heatmap, title, cmap='turbo'):
    signal = np.asarray(signal, dtype=np.float32).flatten()
    heatmap = np.asarray(heatmap, dtype=np.float32).flatten()

    heatmap = gaussian_filter1d(heatmap, sigma=6.0)
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    t = np.arange(len(signal))
    points = np.stack([t, signal], axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    segment_heat = (heatmap[:-1] + heatmap[1:]) / 2

    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(segment_heat)
    lc.set_linewidth(2.5)

    fig, ax = plt.subplots(figsize=(10, 3))
    line = ax.add_collection(lc)
    ax.set_xlim(0, len(signal))

    y_min, y_max = signal.min(), signal.max()
    margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_title(title)

    fig.colorbar(line, ax=ax, label='Importance Score')
    plt.tight_layout()
    filename = f'{title.replace(" ", "_").replace("-", "")}.png'
    plt.savefig(filename, dpi=300)
    print(f"✅ 已保存 -> {filename}")
    plt.show()


def get_1d_gradcam(model, signal, class_idx):
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
    last_conv_layer_name = conv_layers[-1].name

    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    signal_tensor = tf.convert_to_tensor(signal)
    with tf.GradientTape() as tape:
        tape.watch(signal_tensor)
        conv_outputs, predictions = grad_model(signal_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=1)

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap).numpy()

    t_heat = np.linspace(0, 1, len(heatmap))
    t_sig = np.linspace(0, 1, INPUT_LENGTH)
    heatmap_interp = np.interp(t_sig, t_heat, heatmap)

    return heatmap_interp


def get_integrated_gradients(model, signal, class_idx, m_steps=50):
    signal_tensor = tf.convert_to_tensor(signal, dtype=tf.float32)
    signal_single = signal_tensor[0]
    baseline_single = tf.zeros_like(signal_single)

    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    alphas = tf.reshape(alphas, [-1, 1, 1])

    interpolated = baseline_single + alphas * (signal_single - baseline_single)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, interpolated)
    avg_grads = tf.reduce_mean(grads[:-1], axis=0)

    integrated_grad = (signal_single - baseline_single) * avg_grads
    ig_attribution = tf.abs(integrated_grad)
    ig_attribution = tf.squeeze(ig_attribution).numpy()

    return ig_attribution


def get_occlusion_sensitivity(model, signal, class_idx, window_size=25, step=5):
    signal_np = np.asarray(signal)
    seq_len = signal_np.shape[1]

    base_prob = model.predict(signal_np, verbose=0)[0, class_idx]

    sensitivity_map = np.zeros(seq_len)
    overlap_count = np.zeros(seq_len)

    for start in range(0, seq_len, step):
        end = min(start + window_size, seq_len)
        occluded_signal = signal_np.copy()
        occluded_signal[0, start:end, 0] = 0.0

        occ_prob = model.predict(occluded_signal, verbose=0)[0, class_idx]
        prob_drop = base_prob - occ_prob

        sensitivity_map[start:end] += prob_drop
        overlap_count[start:end] += 1

    overlap_count[overlap_count == 0] = 1
    sensitivity_map /= overlap_count

    return sensitivity_map


def analyze_unified_interpretability(model, X, y_true_ohe):
    print("\n[Interpretability] 正在筛选最高置信度的完美样本以固定对比基准...")
    y_true_classes = np.argmax(y_true_ohe, axis=1)

    # ⭐ 修正：使用正确的类别索引
    normal_indices = np.where(y_true_classes == CLASS_NORMAL)[0]
    afib_indices = np.where(y_true_classes == CLASS_AFIB)[0]

    preds = model.predict(X, verbose=0)

    # 找出模型最确信是 Normal 的样本
    normal_scores = preds[normal_indices, CLASS_NORMAL]
    best_normal_idx = normal_indices[np.argmax(normal_scores)]

    # 找出模型最确信是 AFib 的样本
    afib_scores = preds[afib_indices, CLASS_AFIB]
    best_afib_idx = afib_indices[np.argmax(afib_scores)]

    print(f"  -> 固定 Normal 样本索引: {best_normal_idx} (置信度: {preds[best_normal_idx, CLASS_NORMAL]:.4f})")
    print(f"  -> 固定 AFib   样本索引: {best_afib_idx} (置信度: {preds[best_afib_idx, CLASS_AFIB]:.4f})")

    signal_normal = X[best_normal_idx:best_normal_idx + 1]
    signal_afib = X[best_afib_idx:best_afib_idx + 1]

    # ========== 正常心律 (Normal) 分析 ==========
    print("\n[1/2] 正在生成 Normal 样本的可解释性分析...")
    heat_n_cam = get_1d_gradcam(model, signal_normal, class_idx=CLASS_NORMAL)
    plot_saliency_map(signal_normal[0, :, 0], heat_n_cam, "CAM - Normal Sinus Rhythm", cmap='turbo')

    heat_n_ig = get_integrated_gradients(model, signal_normal, class_idx=CLASS_NORMAL)
    plot_saliency_map(signal_normal[0, :, 0], heat_n_ig, "IG - Normal Sinus Rhythm", cmap='magma')

    heat_n_occ = get_occlusion_sensitivity(model, signal_normal, class_idx=CLASS_NORMAL, window_size=25, step=5)
    plot_saliency_map(signal_normal[0, :, 0], heat_n_occ, "Occlusion - Normal Sinus Rhythm", cmap='inferno')

    # ========== 房颤 (AFib) 分析 ==========
    print("\n[2/2] 正在生成 AFib 样本的可解释性分析...")
    heat_a_cam = get_1d_gradcam(model, signal_afib, class_idx=CLASS_AFIB)
    plot_saliency_map(signal_afib[0, :, 0], heat_a_cam, "CAM - Atrial Fibrillation", cmap='turbo')

    heat_a_ig = get_integrated_gradients(model, signal_afib, class_idx=CLASS_AFIB)
    plot_saliency_map(signal_afib[0, :, 0], heat_a_ig, "IG - Atrial Fibrillation", cmap='magma')

    heat_a_occ = get_occlusion_sensitivity(model, signal_afib, class_idx=CLASS_AFIB, window_size=25, step=5)
    plot_saliency_map(signal_afib[0, :, 0], heat_a_occ, "Occlusion - Atrial Fibrillation", cmap='inferno')


# ==========================================
# 6. 主程序调用
# ==========================================
if __name__ == "__main__":
    max_pts = 1500
    if len(X_test_all) > max_pts:
        subset_idx = np.random.choice(len(X_test_all), max_pts, replace=False)
        X_sub = X_test_all[subset_idx]
        y_sub = y_test_all[subset_idx]
    else:
        X_sub = X_test_all
        y_sub = y_test_all

    umap_embedding = plot_umap_representations(model, X_sub, y_sub)
    plot_overlapping_waveforms(X_sub, y_sub, umap_embedding, num_samples=5)
    analyze_unified_interpretability(model, X_sub, y_sub)

    print("\n🎉 所有可视化分析已全部完成！")