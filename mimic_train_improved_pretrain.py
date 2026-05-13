import numpy as np
import pandas as pd
import glob
from scipy.signal import resample, butter, filtfilt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense,
                                     LeakyReLU, BatchNormalization, Dropout, Activation,
                                     Bidirectional, GRU, GlobalAveragePooling1D, Reshape, Multiply)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
import gc
import os
from sklearn.model_selection import GroupShuffleSplit
import warnings

warnings.filterwarnings('ignore')

# =============================================
#  1. 全局配置与 GPU 设置
# =============================================
np.random.seed(42)
tf.random.set_seed(42)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

strategy = tf.distribute.MirroredStrategy()

TRAIN_DATA_DIR = r"D:\AFdataset\mimic_perform_train_all_csv"
MODEL_SAVE_PATH = "mimic_train_best_developed.h5"
INPUT_LENGTH = 800

TARGET_COLUMN = 'PPG'  # 提取的信号列名
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
EPOCHS_PRETRAIN = 50
EPOCHS_FINETUNE = 30


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = K.epsilon() + y_pred
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1.0 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)

    return focal_loss


# =============================================
#  2. 信号处理与质量控制 (核心优化部分)
# =============================================
def butter_bandpass_filter(data, lowcut=0.5, highcut=18.0, fs=125, order=2):
    """0.5Hz过滤基线漂移，18.0Hz过滤高频噪声"""
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
    if np.std(signal) < 0.05:
        return False

    # 3. 🛡️ 核心新增：拦截剧烈运动伪影和基线突变
    # 使用 IQR (四分位距) 衡量信号的主体波动幅度
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25

    # 防止除以0
    if iqr < 1e-5:
        return False

    # 生理信号的极值通常不会超过主体波动幅度(IQR)的太多倍。
    # 如果最大值超出中位数 4 倍以上的 IQR，说明这是一个因为身体剧烈晃动产生的巨大尖峰！
    max_dev = (np.max(signal) - np.median(signal)) / iqr
    min_dev = (np.median(signal) - np.min(signal)) / iqr

    if max_dev > 4.0 or min_dev > 4.0:
        return False

    return True


def load_training_data(csv_dir, input_length=INPUT_LENGTH, target_col=TARGET_COLUMN):
    print(f"\n[Data Load] 正在从 {csv_dir} 加载训练数据与标签...")
    excel_files = glob.glob(os.path.join(csv_dir, "**", "*_data.csv*"), recursive=True)
    if len(excel_files) == 0:
        raise ValueError(f"❌ 未找到任何 *_data.csv 文件！请检查路径：{csv_dir}")

    all_signals, all_labels, all_groups = [], [], []
    success_files, missing_txt, discarded_chunks = 0, 0, 0

    for file_idx, excel_path in enumerate(excel_files):
        dir_name = os.path.dirname(excel_path)
        file_name = os.path.basename(excel_path)

        prefix = file_name.split('_data')[0]
        txt_name = prefix + "_fix.txt"
        txt_path = os.path.join(dir_name, txt_name)

        if not os.path.exists(txt_path):
            missing_txt += 1
            continue

        # 解析 TXT 中的标签
        label = None
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith("Group:"):
                        val = line.split(":")[1].strip().lower()
                        if val == 'n':
                            label = [1.0, 0.0]  # Normal
                        elif val == 'a':
                            label = [0.0, 1.0]  # AF
                        break
        except Exception:
            continue

        if label is None: continue

        # 读取 Excel 信号并切片
        try:
            df = pd.read_csv(excel_path, encoding='utf-8')
            df.columns = df.columns.astype(str).str.strip()
            if target_col not in df.columns: continue

            sig = np.nan_to_num(pd.to_numeric(df[target_col].values, errors='coerce'))
            sig = butter_bandpass_filter(sig)  # 带通滤波

            num_chunks = len(sig) // input_length
            for i in range(num_chunks):
                chunk = sig[i * input_length: (i + 1) * input_length]
                chunk_mean = np.mean(chunk)
                chunk_std = np.std(chunk)
                if chunk_std == 0: chunk_std = 1.0
                chunk = (chunk - chunk_mean) / chunk_std

                # ⭐ 质量检验：拦截垃圾波形
                if is_signal_valid(chunk):
                    all_signals.append(chunk)
                    all_labels.append(label)
                    all_groups.append(file_idx)
                else:
                    discarded_chunks += 1

            success_files += 1

        except Exception as e:
            pass

    print(f"✅ 解析完成！成功读取 {success_files} 名患者数据。")
    print(f"🛡️ SQA质量控制：已自动丢弃 {discarded_chunks} 个低质量/损坏片段。")
    print(f"📊 最终可用高质量切片: {len(all_signals)} 个。")

    if len(all_signals) == 0:
        raise ValueError("❌ 可用数据为0！可能是阈值设置过高或数据源问题。")

    return np.array(all_signals, dtype=np.float32), np.array(all_labels, dtype=np.float32), np.array(all_groups)


# =============================================
#  3. 数据生成器与模型定义
# =============================================
class DataGenerator(keras.utils.Sequence):
    def __init__(self, signals, labels, indices, batch_size=32, shuffle=True, class_weights=None):
        self.signals = signals
        self.labels = labels
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.signals[batch_indices]
        y = self.labels[batch_indices]

        # ⭐ 修复：计算并返回 sample_weights
        if self.class_weights is not None:
            y_classes = np.argmax(y, axis=1)
            sample_weights = np.array([self.class_weights[c] for c in y_classes])
            return X, y, sample_weights

        return X, y


def gen_ppg_pulse():
    return np.array(
        [1077, 1095, 1150, 1253, 1410, 1614, 1852, 2107, 2354, 2571, 2747, 2878, 2967, 3018, 3042, 3047, 3037, 3018,
         2991, 2959, 2924, 2885, 2843, 2802, 2759, 2716, 2673, 2630, 2584, 2533, 2480, 2423, 2360, 2294, 2226, 2155,
         2085, 2019, 1959, 1905, 1860, 1823, 1799, 1785, 1781, 1785, 1795, 1810, 1825, 1843, 1863, 1877, 1882, 1880,
         1874, 1862, 1847, 1829, 1810, 1788, 1765, 1742, 1717, 1692, 1668, 1644, 1619, 1595, 1572, 1548, 1525, 1501,
         1478, 1455, 1433, 1412, 1390, 1369, 1348, 1328, 1308, 1289, 1272, 1256, 1241, 1229, 1217, 1206, 1195, 1185,
         1176, 1169, 1160, 1153, 1146, 1138, 1127, 1112, 1097, 1083, 1080], dtype=np.float32)


def generate_simulated_batch(num_samples=1000, duration=25, fs=32):
    clean_sigs, noisy_sigs = [], []
    temp_v = gen_ppg_pulse()
    temp_v = (temp_v - np.mean(temp_v)) / (np.std(temp_v) + 1e-7)

    total_pts = int(duration * fs * 4)
    for _ in range(num_samples):
        is_af = np.random.rand() > 0.5
        hr = np.random.uniform(60, 100)
        sig = []
        while len(sig) < total_pts:
            rr = 60.0 / hr
            if is_af: rr = np.clip(rr * np.random.normal(1.0, 0.2), 0.3, 1.5)
            beat_len = int(rr * fs * 4)
            if beat_len > 0: sig.extend(resample(temp_v, beat_len))

        sig = np.array(sig[:total_pts])
        s_res = resample(sig, int(duration * fs))

        s_clean = (s_res - np.mean(s_res)) / (np.std(s_res) + 1e-7)
        noise = np.random.normal(0, np.random.choice([0.1, 0.5, 1.0]), len(s_clean))
        s_noisy = s_clean + noise
        s_noisy = (s_noisy - np.mean(s_noisy)) / (np.std(s_noisy) + 1e-7)

        clean_sigs.append(s_clean.reshape(-1, 1))
        noisy_sigs.append(s_noisy.reshape(-1, 1))
    return np.array(clean_sigs, dtype=np.float32), np.array(noisy_sigs, dtype=np.float32)


def get_enc(x):
    for f in [16, 32, 64]:
        x = Conv1D(f, 5, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same')(x)
    return x


def define_cdae():
    inp = Input(shape=(INPUT_LENGTH, 1))
    x = get_enc(inp)
    for f in [64, 32, 16]:
        x = Conv1D(f, 5, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2)(x)

    out = Conv1D(1, 5, padding='same', activation='linear', dtype='float32')(x)
    return Model(inp, out)


def attention_block(inputs):
    input_channels = inputs.shape[-1]
    x = GlobalAveragePooling1D()(inputs)
    x = Dense(input_channels // 4, activation='relu')(x)
    x = Dense(input_channels, activation='sigmoid')(x)
    x = Reshape((1, input_channels))(x)
    return Multiply()([inputs, x])


def define_af_net():
    inputs = Input(shape=(INPUT_LENGTH, 1))

    # --- 第一层：多尺度特征提取 ---
    x = Conv1D(32, kernel_size=15, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv1D(64, kernel_size=9, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # --- 第二层：卷积 + 注意力机制 ---
    x = Conv1D(128, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = attention_block(x)  # 引入注意力
    x = MaxPooling1D(pool_size=2)(x)

    # --- 第三层：循环神经网络层 (RNN) ---
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)

    # --- 全连接输出层 ---
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    outputs = Dense(2, activation='softmax', name='rhythm')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# =============================================
#  4. 执行主流程
# =============================================
def main():
    print("\n" + "=" * 45)
    print(" 🎯 模型训练 (真实临床数据 Training)")
    print("=" * 45)

    try:
        signals, labels, groups = load_training_data(TRAIN_DATA_DIR)
    except Exception as e:
        print(e)
        return

    all_idx = np.arange(len(signals))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    t_idx, v_idx = next(gss.split(all_idx, groups=groups))

    n_normal, n_af = np.sum(labels[t_idx][:, 0] == 1), np.sum(labels[t_idx][:, 1] == 1)
    print(f"✅ 数据划分完毕 -> 训练集片段分布: 正常(0): {n_normal} | 房颤(1): {n_af}")

    # ⭐ 修复：优化权重分配，让正常的权重稍微大一点点，抑制假阳性
    tot = n_normal + n_af
    cw = {
        0: (1 / max(n_normal, 1)) * (tot / 2.0) * 1.2,
        1: (1 / max(n_af, 1)) * (tot / 2.0) * 1.0
    }
    print(f"⚖️ 使用类别权重: {cw}")

    t_gen = DataGenerator(signals, labels, t_idx, BATCH_SIZE, shuffle=True, class_weights=cw)
    v_gen = DataGenerator(signals, labels, v_idx, BATCH_SIZE, shuffle=False)

    with strategy.scope():
        model = define_af_net()
        # 使用带有一点标签平滑的 Loss 防止过度自信
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

        model.compile(
            optimizer=Adam(learning_rate=3e-4, clipnorm=1.0),
            loss=loss_fn,
            metrics=['accuracy']
        )

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    model.fit(t_gen, validation_data=v_gen, epochs=EPOCHS_FINETUNE, callbacks=callbacks)
    model.save(MODEL_SAVE_PATH)
    print(f"\n🎉 完美！模型训练完成并已保存至: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()