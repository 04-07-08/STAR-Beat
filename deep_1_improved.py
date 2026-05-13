import numpy as np
from scipy.signal import detrend, resample
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, LeakyReLU, \
    BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
import gc
import sys
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Bidirectional, GRU, GlobalAveragePooling1D, Reshape, Multiply
import tensorflow.keras.backend as K

# =============================================
#  1. GPU 加速配置
# =============================================
# 固定随机种子
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Found {len(gpus)} GPU(s): {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU found! Training will be slow.")

# ⚠️ 暂时关闭混合精度以防止 Loss NaN
# if gpus:
#    policy = mixed_precision.Policy('mixed_float16')
#    mixed_precision.set_global_policy(policy)

# 定义策略
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices in strategy: {strategy.num_replicas_in_sync}")

# =============================================
#  配置与全局变量
# =============================================
# 请确保此处路径正确
REAL_DATA_PATH = r"D:\AFdataset\train.npz"
EXTRACTED_DIR = r"D:\AFdataset\train_extracted"
MODEL_SAVE_PATH = "deepbeat_finetuned_improved_best_1.h5"  # 修改文件名，强调是Best

INPUT_LENGTH = 800
BATCH_SIZE = 256 * strategy.num_replicas_in_sync
EPOCHS_PRETRAIN = 50
EPOCHS_FINETUNE = 30  # 可以设大一点，依靠 EarlyStopping 和 Save Best 自动停止


# =============================================
#  数据生成器 (含安全归一化)
# =============================================
class DataGenerator(keras.utils.Sequence):
    def __init__(self, signals, qa_labels, rhythm_labels, indices, batch_size=128, shuffle=True, class_weights=None):
        self.signals = signals
        self.qa_labels = qa_labels
        self.rhythm_labels = rhythm_labels
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_weights = class_weights
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.signals[batch_indices].astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        # 创新 1：去除基线漂移 (Detrending)
        # axis=1 确保我们在每个样本的时间轴上进行操作
        X = detrend(X, type='linear', axis=1)

        # 创新 2：使用 Z-score 标准化替代 Min-Max
        mean_val = np.mean(X, axis=1, keepdims=True)
        std_val = np.std(X, axis=1, keepdims=True)
        # 防止除以 0
        std_val[std_val == 0] = 1.0
        X = (X - mean_val) / std_val

        # 二次检查，确保没有计算产生的 NaN
        X = np.nan_to_num(X, nan=0.0)

        if X.ndim == 2:
            X = X[..., np.newaxis]


        y_qa = self.qa_labels[batch_indices].astype(np.float32)
        y_rhythm = self.rhythm_labels[batch_indices].astype(np.float32)

        # 样本权重计算
        sample_weights = None
        if self.class_weights:
            sample_weights = {}
            qa_classes = np.argmax(y_qa, axis=1)
            qa_sw = np.array([self.class_weights['qa'].get(c, 1.0) for c in qa_classes])
            sample_weights['qa'] = qa_sw

            r_classes = np.argmax(y_rhythm, axis=1)
            r_sw = np.array([self.class_weights['rhythm'].get(c, 1.0) for c in r_classes])
            sample_weights['rhythm'] = r_sw

        return X, {'qa': y_qa, 'rhythm': y_rhythm}, sample_weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================
#  PPG 模拟信号生成 (CPU)
# =============================================
def gen_ppg_pulse():
    ppg_v = np.array([
        1077, 1095, 1150, 1253, 1410, 1614, 1852, 2107, 2354, 2571,
        2747, 2878, 2967, 3018, 3042, 3047, 3037, 3018, 2991, 2959,
        2924, 2885, 2843, 2802, 2759, 2716, 2673, 2630, 2584, 2533,
        2480, 2423, 2360, 2294, 2226, 2155, 2085, 2019, 1959, 1905,
        1860, 1823, 1799, 1785, 1781, 1785, 1795, 1810, 1825, 1843,
        1863, 1877, 1882, 1880, 1874, 1862, 1847, 1829, 1810, 1788,
        1765, 1742, 1717, 1692, 1668, 1644, 1619, 1595, 1572, 1548,
        1525, 1501, 1478, 1455, 1433, 1412, 1390, 1369, 1348, 1328,
        1308, 1289, 1272, 1256, 1241, 1229, 1217, 1206, 1195, 1185,
        1176, 1169, 1160, 1153, 1146, 1138, 1127, 1112, 1097, 1083, 1080
    ], dtype=np.float32)
    ppg_t = np.linspace(0, 1, len(ppg_v))
    return ppg_t, ppg_v


def generate_simulated_batch(num_samples=1000, duration=25, fs=32):
    # 为节省篇幅，使用原有逻辑，此处略微简化以确保可运行
    clean_signals = []
    noisy_signals = []
    noise_levels = [0.001, 0.15, 0.5, 0.75, 1.0, 2.0]
    template_t, template_v = gen_ppg_pulse()
    template_v = (template_v - np.min(template_v)) / (np.max(template_v) - np.min(template_v))
    gen_fs = fs * 4
    total_points = int(duration * gen_fs)

    for _ in range(num_samples):
        is_af = np.random.rand() > 0.5
        hr_mean = np.random.uniform(60, 100)
        sig = []
        # 简化版生成逻辑
        while len(sig) < total_points:
            rr_interval = 60.0 / hr_mean
            if is_af:
                rr_interval *= np.random.normal(1.0, 0.2)
                rr_interval = np.clip(rr_interval, 0.3, 1.5)
            beat_len = int(rr_interval * gen_fs)
            if beat_len > 0:
                beat = resample(template_v, beat_len)
                sig.extend(beat)
        sig = np.array(sig[:total_points])
        sig_resampled = resample(sig, int(duration * fs))
        sig_clean = (sig_resampled - np.min(sig_resampled)) / (np.max(sig_resampled) - np.min(sig_resampled) + 1e-7)
        sigma = np.random.choice(noise_levels)
        noise = np.random.normal(0, sigma, len(sig_clean))
        sig_noisy = sig_clean + noise
        sig_noisy = (sig_noisy - np.min(sig_noisy)) / (np.max(sig_noisy) - np.min(sig_noisy) + 1e-7)
        clean_signals.append(sig_clean.reshape(-1, 1))
        noisy_signals.append(sig_noisy.reshape(-1, 1))
    return np.array(clean_signals, dtype=np.float32), np.array(noisy_signals, dtype=np.float32)


# =============================================
#  模型定义
# =============================================
def get_encoder_layers(input_tensor):
    x = Conv1D(16, 5, padding='same', kernel_initializer='he_normal', name='enc_conv1')(input_tensor)
    x = BatchNormalization(name='enc_bn1')(x)
    x = Activation('relu', name='enc_act1')(x)
    x = MaxPooling1D(2, padding='same', name='enc_pool1')(x)
    x = Conv1D(32, 5, padding='same', kernel_initializer='he_normal', name='enc_conv2')(x)
    x = BatchNormalization(name='enc_bn2')(x)
    x = Activation('relu', name='enc_act2')(x)
    x = MaxPooling1D(2, padding='same', name='enc_pool2')(x)
    x = Conv1D(64, 5, padding='same', kernel_initializer='he_normal', name='enc_conv3')(x)
    x = BatchNormalization(name='enc_bn3')(x)
    x = Activation('relu', name='enc_act3')(x)
    encoded = MaxPooling1D(2, padding='same', name='enc_pool3')(x)
    return encoded


def define_cdae():
    input_sig = Input(shape=(INPUT_LENGTH, 1), name='input_signal')
    encoded = get_encoder_layers(input_sig)
    # Decoder
    x = Conv1D(64, 5, padding='same', kernel_initializer='he_normal')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 5, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(1, 5, padding='same', name='reconstruction_logits')(x)
    decoded = Activation('sigmoid', dtype='float32', name='reconstruction')(x)

    cdae = Model(input_sig, decoded, name='CDAE')
    cdae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return cdae


def se_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation Block 用于通道注意力机制"""
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    # 将其 reshape 为 (1, filters) 以便与 (batch, steps, filters) 广播相乘
    se = Reshape((1, filters))(se)
    return Multiply()([input_tensor, se])

def define_deepbeat(pretrained_encoder_weights=None):
    input_sig = Input(shape=(INPUT_LENGTH, 1), name='input_signal')

    # --- Encoder (保持与 CDAE 一致，以便加载预训练权重) ---
    x = Conv1D(16, 5, padding='same', kernel_initializer='he_normal', name='enc_conv1')(input_sig)
    x = BatchNormalization(name='enc_bn1')(x)
    x = Activation('relu', name='enc_act1')(x)
    x = MaxPooling1D(2, padding='same', name='enc_pool1')(x)

    x = Conv1D(32, 5, padding='same', kernel_initializer='he_normal', name='enc_conv2')(x)
    x = BatchNormalization(name='enc_bn2')(x)
    x = Activation('relu', name='enc_act2')(x)
    x = MaxPooling1D(2, padding='same', name='enc_pool2')(x)

    x = Conv1D(64, 5, padding='same', kernel_initializer='he_normal', name='enc_conv3')(x)
    x = BatchNormalization(name='enc_bn3')(x)
    x = Activation('relu', name='enc_act3')(x)
    encoded = MaxPooling1D(2, padding='same', name='enc_pool3')(x)

    # --- Shared Layers (引入 SE Block) ---
    x = Conv1D(128, 5, padding='same', kernel_initializer='he_normal', name='shared_conv4')(encoded)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = se_block(x) # 引入注意力
    x = Dropout(0.25)(x)

    x = Conv1D(256, 5, padding='same', kernel_initializer='he_normal', name='shared_conv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = se_block(x) # 引入注意力
    x = Dropout(0.25)(x)

    x = Conv1D(512, 5, padding='same', kernel_initializer='he_normal', name='shared_conv6')(x)
    x = BatchNormalization()(x)
    shared_features = LeakyReLU(alpha=0.1)(x)
    shared_features = Dropout(0.25)(shared_features)

    # --- QA Branch (质量评估侧重于全局波形，保持 CNN + 全局池化) ---
    # 使用 GlobalAveragePooling1D 替代 Flatten 可以大幅减少参数量，降低过拟合风险
    qa_x = GlobalAveragePooling1D()(shared_features)
    qa_x = Dense(128, activation='relu')(qa_x)
    qa_x = Dropout(0.5)(qa_x)
    qa_x = Dense(64, activation='relu')(qa_x)
    qa_logits = Dense(3, name='qa_logits')(qa_x)
    qa_out = Activation('softmax', dtype='float32', name='qa')(qa_logits)

    # --- Rhythm Branch (节律评估侧重于时序关联，引入 BiGRU) ---
    r_x = Conv1D(256, 3, padding='same', activation='relu')(shared_features)
    r_x = BatchNormalization()(r_x)
    r_x = MaxPooling1D(2)(r_x)
    # 使用 BiGRU 捕获长序列中的心跳间隔（RR间期）变化
    r_x = Bidirectional(GRU(64, return_sequences=False))(r_x)
    r_x = Dense(128, activation='relu')(r_x)
    r_x = Dropout(0.5)(r_x)
    r_x = Dense(64, activation='relu')(r_x)
    rhythm_logits = Dense(2, name='rhythm_logits')(r_x)
    rhythm_out = Activation('softmax', dtype='float32', name='rhythm')(rhythm_logits)

    model = Model(inputs=input_sig, outputs=[qa_out, rhythm_out], name='DeepBeat_Improved')

    if pretrained_encoder_weights:
        print("Transferring pretrained CDAE weights...")
        layer_names = ['enc_conv1', 'enc_bn1', 'enc_conv2', 'enc_bn2', 'enc_conv3', 'enc_bn3']
        for name in layer_names:
            try:
                model.get_layer(name).set_weights(pretrained_encoder_weights[name])
            except Exception as e:
                print(f" ! Failed to load layer {name}: {e}")
    return model

# 定义Focal Loss
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    带 Alpha 权重的分类 Focal Loss
    alpha 负责处理类别不平衡，gamma 负责处理难易样本不平衡
    """
    def focal_loss(y_true, y_pred):
        # 裁剪预测值防止 log(0) 导致的 NaN
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        # 计算交叉熵
        cross_entropy = -y_true * K.log(y_pred)
        # 计算调节系数 (1 - pt)^gamma
        weight = alpha * K.pow(1.0 - y_pred, gamma)
        # 计算最终 Loss
        loss = weight * cross_entropy
        return K.sum(loss, axis=-1)
    return focal_loss

# =============================================
#  主流程
# =============================================
def main():
    # -------------------------------------
    # Phase 1: CDAE Pretraining
    # -------------------------------------
    print("\n" + "=" * 40)
    print(" Phase 1: CDAE Pretraining (Simulation)")
    print("=" * 40)

    N_SIM = 30000
    print(f"Generating {N_SIM} simulated samples...")
    clean_sim, noisy_sim = generate_simulated_batch(N_SIM, duration=25, fs=32)
    clean_val, noisy_val = generate_simulated_batch(2000, duration=25, fs=32)

    with strategy.scope():
        cdae = define_cdae()

    es_cdae = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    cdae.fit(
        noisy_sim, clean_sim,
        validation_data=(noisy_val, clean_val),
        epochs=EPOCHS_PRETRAIN,
        batch_size=BATCH_SIZE,
        callbacks=[es_cdae],
        verbose=1
    )

    print("Extracting encoder weights...")
    pretrained_weights = {}
    layer_names = ['enc_conv1', 'enc_bn1', 'enc_conv2', 'enc_bn2', 'enc_conv3', 'enc_bn3']
    for name in layer_names:
        pretrained_weights[name] = cdae.get_layer(name).get_weights()

    del cdae, clean_sim, noisy_sim, clean_val, noisy_val
    keras.backend.clear_session()
    gc.collect()

    # -------------------------------------
    # Phase 2: DeepBeat Fine-tuning
    # -------------------------------------
    print("\n" + "=" * 40)
    print(" Phase 2: DeepBeat Fine-tuning (Real Data)")
    print("=" * 40)

    if not os.path.exists(EXTRACTED_DIR):
        print(f"Error: 文件夹不存在 {EXTRACTED_DIR}")
        print("请先运行 extract_data.py 解压数据！")
        return

    print("Loading real dataset from .npy files...")
    signals_path = os.path.join(EXTRACTED_DIR, 'signal.npy')
    qa_path = os.path.join(EXTRACTED_DIR, 'qa_label.npy')
    rhythm_path = os.path.join(EXTRACTED_DIR, 'rhythm.npy')
    params_path = os.path.join(EXTRACTED_DIR, 'parameters.npy')

    if not os.path.exists(signals_path):
        print(f"Error: 找不到 {signals_path}")
        return

    # mmap 读取大文件
    signals = np.load(signals_path, mmap_mode='r')
    qa_labels = np.load(qa_path, mmap_mode='r')
    rhythm_labels = np.load(rhythm_path, mmap_mode='r')
    p_data = np.load(params_path, allow_pickle=True)
    parameters = pd.DataFrame(p_data, columns=['timestamp', 'stream', 'ID'])

    print(f"Total samples: {len(parameters)}")
    all_indices = np.arange(len(parameters))

    # 按病人划分训练集/验证集
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(all_indices, groups=parameters['ID']))

    # --------------------------------------------------------
    # 核心修改：手动设置类别权重，防止过拟合/过于敏感
    # --------------------------------------------------------
    print("Computing class weights...")

    # 1. QA 权重: 优(Class 2) 是少数，适当加权
    qa_cw_dict = {0: 1.0, 1: 1.0, 2: 2.0}

    # 2. Rhythm 权重:
    # 之前的 Balanced 可能导致 AFib 权重过大 (如 1:20)，导致 TPR=1.0 但 PPV 低
    # 这里改为 {0: 1.0, 1: 3.0}。
    # 含义：漏掉一个 AFib (1) 的惩罚是漏掉 Normal (0) 的 3 倍。
    # 如果这依然导致过敏(TPR=1.0)，可改为 {0: 1.0, 1: 2.0} 甚至 {0: 1.0, 1: 1.0}
    r_cw_dict = {0: 3.0, 1: 1.0}

    print(f"Used QA Weights: {qa_cw_dict}")
    print(f"Used Rhythm Weights: {r_cw_dict}")

    weight_maps = {'qa': qa_cw_dict, 'rhythm': r_cw_dict}

    current_bs = BATCH_SIZE
    print(f"Training with Batch Size: {current_bs}")

    train_gen = DataGenerator(signals, qa_labels, rhythm_labels, train_idx, batch_size=current_bs, shuffle=True,
                              class_weights=weight_maps)
    val_gen = DataGenerator(signals, qa_labels, rhythm_labels, val_idx, batch_size=current_bs, shuffle=False)

    with strategy.scope():
        model = define_deepbeat(pretrained_weights)  # 使用改进的网络
        opt = Adam(learning_rate=1e-5, clipnorm=1.0)

        model.compile(
            optimizer=opt,
            # 使用自定义的 Focal loss 替代常规的 categorical_crossentropy
            loss={'qa': categorical_focal_loss(gamma=2.0, alpha=0.25),
                  'rhythm': categorical_focal_loss(gamma=2.0, alpha=0.25)},
            loss_weights={'qa': 1.0, 'rhythm': 1.0},
            metrics={'qa': 'accuracy', 'rhythm': 'accuracy'},
        )

    # --------------------------------------------------------
    # 核心修改：回调函数，锁定最佳模型
    # --------------------------------------------------------
    # 1. 监控 val_loss 而不是 accuracy，防止过拟合
    # 2. save_best_only=True 确保保存的是 Epoch 3 (如果它最好的话)
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)

    # 3. restore_best_weights=True 确保训练结束后 model 回滚到最佳状态
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    print("\nStarting Fine-tuning...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1,
        workers=8,
        use_multiprocessing=False  # Windows下必须为False
    )

    # 这里的 save 保存的是 restore 回来的最佳权重
    model.save(MODEL_SAVE_PATH)
    print(f"\nBest Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()