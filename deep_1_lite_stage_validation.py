import numpy as np
from scipy.signal import detrend, resample
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, LeakyReLU, \
    BatchNormalization, Dropout, Activation, SeparableConv1D, GlobalAveragePooling1D, Reshape, Multiply, Bidirectional, \
    GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import gc
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tensorflow.keras.backend as K

# =====================================================================
# 🚀 核心控制台：消融实验控制开关
# =====================================================================
# 修改这个数字来测试不同的轻量化阶段：
# 0 = Baseline (原始 improved 版本，参数量最大，包含大卷积核和 BiGRU)
# 1 = Baseline + 共享层轻量化 (SeparableConv & 通道减半)
# 2 = 阶段 1 + 分类器轻量化 (Dense 层节点减半，Dropout 降低)
# 3 = 阶段 2 + 移除 BiGRU (最终的 Lite 版本，极速推理)
# =====================================================================
ABLATION_STEP = 2  # ⬅️ 每次做消融实验前，修改这里！

# 模型保存名称会根据阶段自动调整
MODEL_SAVE_PATH = f"deepbeat_ablation_step_{ABLATION_STEP}.h5"

# =============================================
#  1. 全局与 GPU 配置
# =============================================
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

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices in strategy: {strategy.num_replicas_in_sync}")

EXTRACTED_DIR = r"D:\AFdataset\train_extracted"
INPUT_LENGTH = 800
BATCH_SIZE = 256 * strategy.num_replicas_in_sync
EPOCHS_PRETRAIN = 50
EPOCHS_FINETUNE = 30  # 依靠 EarlyStopping 自动停止


# =============================================
#  2. 数据生成器 (DataGenerator)
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

        # 归一化：去基线漂移 (Detrending) + Z-score
        X = detrend(X, type='linear', axis=1)
        mean_val = np.mean(X, axis=1, keepdims=True)
        std_val = np.std(X, axis=1, keepdims=True)
        std_val[std_val == 0] = 1.0
        X = (X - mean_val) / std_val
        X = np.nan_to_num(X, nan=0.0)

        if X.ndim == 2:
            X = X[..., np.newaxis]

        y_qa = self.qa_labels[batch_indices].astype(np.float32)
        y_rhythm = self.rhythm_labels[batch_indices].astype(np.float32)

        sample_weights = None
        if self.class_weights:
            sample_weights = {}
            qa_sw = np.array([self.class_weights['qa'].get(c, 1.0) for c in np.argmax(y_qa, axis=1)])
            r_sw = np.array([self.class_weights['rhythm'].get(c, 1.0) for c in np.argmax(y_rhythm, axis=1)])
            sample_weights['qa'], sample_weights['rhythm'] = qa_sw, r_sw

        return X, {'qa': y_qa, 'rhythm': y_rhythm}, sample_weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================
#  3. 损失函数定义
# =============================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1.0 - y_pred, gamma)
        return K.sum(weight * cross_entropy, axis=-1)

    return focal_loss


# =============================================
#  4. 模拟信号生成 (用于 Phase 1 预训练)
# =============================================
def gen_ppg_pulse():
    ppg_v = np.array([
        1077, 1095, 1150, 1253, 1410, 1614, 1852, 2107, 2354, 2571, 2747, 2878, 2967, 3018, 3042, 3047,
        3037, 3018, 2991, 2959, 2924, 2885, 2843, 2802, 2759, 2716, 2673, 2630, 2584, 2533, 2480, 2423,
        2360, 2294, 2226, 2155, 2085, 2019, 1959, 1905, 1860, 1823, 1799, 1785, 1781, 1785, 1795, 1810,
        1825, 1843, 1863, 1877, 1882, 1880, 1874, 1862, 1847, 1829, 1810, 1788, 1765, 1742, 1717, 1692,
        1668, 1644, 1619, 1595, 1572, 1548, 1525, 1501, 1478, 1455, 1433, 1412, 1390, 1369, 1348, 1328,
        1308, 1289, 1272, 1256, 1241, 1229, 1217, 1206, 1195, 1185, 1176, 1169, 1160, 1153, 1146, 1138,
        1127, 1112, 1097, 1083, 1080
    ], dtype=np.float32)
    ppg_t = np.linspace(0, 1, len(ppg_v))
    return ppg_t, ppg_v


def generate_simulated_batch(num_samples=1000, duration=25, fs=32):
    clean_signals, noisy_signals = [], []
    noise_levels = [0.001, 0.15, 0.5, 0.75, 1.0, 2.0]
    template_t, template_v = gen_ppg_pulse()
    template_v = (template_v - np.min(template_v)) / (np.max(template_v) - np.min(template_v))
    gen_fs = fs * 4
    total_points = int(duration * gen_fs)

    for _ in range(num_samples):
        is_af = np.random.rand() > 0.5
        hr_mean = np.random.uniform(60, 100)
        sig = []
        while len(sig) < total_points:
            rr_interval = 60.0 / hr_mean
            if is_af:
                rr_interval *= np.random.normal(1.0, 0.2)
                rr_interval = np.clip(rr_interval, 0.3, 1.5)
            beat_len = int(rr_interval * gen_fs)
            if beat_len > 0:
                sig.extend(resample(template_v, beat_len))

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
#  5. 模型架构定义区 (消融实验核心)
# =============================================
def se_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation Block"""
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    se = Reshape((1, filters))(se)
    return Multiply()([input_tensor, se])


def get_encoder_layers(input_tensor):
    """通用的编码器层，确保与预训练网络兼容"""
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
    """定义去噪自编码器，用于 Phase 1 预训练"""
    input_sig = Input(shape=(INPUT_LENGTH, 1), name='input_signal')
    encoded = get_encoder_layers(input_sig)

    x = Conv1D(64, 5, padding='same', kernel_initializer='he_normal')(encoded)
    x = BatchNormalization()(x);
    x = Activation('relu')(x);
    x = UpSampling1D(2)(x)

    x = Conv1D(32, 5, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x);
    x = Activation('relu')(x);
    x = UpSampling1D(2)(x)

    x = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x);
    x = Activation('relu')(x);
    x = UpSampling1D(2)(x)

    x = Conv1D(1, 5, padding='same', name='reconstruction_logits')(x)
    decoded = Activation('sigmoid', dtype='float32', name='reconstruction')(x)

    cdae = Model(input_sig, decoded, name='CDAE')
    cdae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return cdae


def build_ablation_model(step, pretrained_weights=None):
    """
    ✨ 核心函数：根据所选消融实验阶段动态生成网络结构
    """
    input_sig = Input(shape=(INPUT_LENGTH, 1), name='input_signal')
    encoded = get_encoder_layers(input_sig)

    # ----------------------------------------------------
    # 操作 A: 共享特征提取层轻量化
    # ----------------------------------------------------
    if step >= 1:
        # 轻量化模式 (SeparableConv1D + 通道减半)
        x = SeparableConv1D(64, 5, padding='same', name='shared_sepconv4')(encoded)
        x = BatchNormalization()(x);
        x = LeakyReLU(alpha=0.1)(x);
        x = se_block(x);
        x = Dropout(0.2)(x)

        x = SeparableConv1D(128, 5, padding='same', name='shared_sepconv5')(x)
        x = BatchNormalization()(x);
        x = LeakyReLU(alpha=0.1)(x);
        x = se_block(x);
        x = Dropout(0.2)(x)

        x = SeparableConv1D(256, 5, padding='same', name='shared_sepconv6')(x)
        x = BatchNormalization()(x)
        shared_features = LeakyReLU(alpha=0.1)(x)
    else:
        # 原始基线模式 (常规 Conv1D + 大通道)
        x = Conv1D(128, 5, padding='same', kernel_initializer='he_normal', name='shared_conv4')(encoded)
        x = BatchNormalization()(x);
        x = LeakyReLU(alpha=0.1)(x);
        x = se_block(x);
        x = Dropout(0.25)(x)

        x = Conv1D(256, 5, padding='same', kernel_initializer='he_normal', name='shared_conv5')(x)
        x = BatchNormalization()(x);
        x = LeakyReLU(alpha=0.1)(x);
        x = se_block(x);
        x = Dropout(0.25)(x)

        x = Conv1D(512, 5, padding='same', kernel_initializer='he_normal', name='shared_conv6')(x)
        x = BatchNormalization()(x)
        shared_features = LeakyReLU(alpha=0.1)(x)
        shared_features = Dropout(0.25)(shared_features)

    # ----------------------------------------------------
    # 操作 B: 全连接分类器层瘦身配置
    # ----------------------------------------------------
    dense_dim_1 = 64 if step >= 2 else 128
    dense_dim_2 = 32 if step >= 2 else 64
    drop_rate = 0.3 if step >= 2 else 0.5

    # QA 分支始终使用 GAP 替代 Flatten
    qa_x = GlobalAveragePooling1D()(shared_features)
    qa_x = Dense(dense_dim_1, activation='relu')(qa_x)
    qa_x = Dropout(drop_rate)(qa_x)
    qa_x = Dense(dense_dim_2, activation='relu')(qa_x)
    qa_out = Dense(3, activation='softmax', name='qa')(qa_x)

    # ----------------------------------------------------
    # 操作 C: Rhythm 分支结构
    # ----------------------------------------------------
    if step >= 3:
        # Lite 模式：移除 BiGRU，纯卷积池化
        r_x = SeparableConv1D(128, 3, padding='same', activation='relu')(shared_features)
        r_x = BatchNormalization()(r_x)
        r_x = MaxPooling1D(2)(r_x)
        r_x = GlobalAveragePooling1D()(r_x)
    else:
        # 基线模式：保留耗时的 BiGRU
        conv_filters = 128 if step >= 1 else 256
        r_x = Conv1D(conv_filters, 3, padding='same', activation='relu')(shared_features)
        r_x = BatchNormalization()(r_x)
        r_x = MaxPooling1D(2)(r_x)
        r_x = Bidirectional(GRU(64, return_sequences=False))(r_x)

        # Rhythm 后续分类网络 (应用 操作 B 配置)
    r_x = Dense(dense_dim_1, activation='relu')(r_x)
    r_x = Dropout(drop_rate)(r_x)
    r_x = Dense(dense_dim_2, activation='relu')(r_x)
    rhythm_out = Dense(2, activation='softmax', name='rhythm')(r_x)

    model = Model(inputs=input_sig, outputs=[qa_out, rhythm_out], name=f'DeepBeat_Ablation_Step_{step}')

    # 打印参数量信息以便观察轻量化效果
    model.summary()

    # 载入 Phase 1 预训练权重
    if pretrained_weights:
        print("Transferring pretrained CDAE encoder weights...")
        for name in ['enc_conv1', 'enc_bn1', 'enc_conv2', 'enc_bn2', 'enc_conv3', 'enc_bn3']:
            try:
                model.get_layer(name).set_weights(pretrained_weights[name])
            except Exception as e:
                print(f" ! Failed to load {name}: {e}")

    return model


# =============================================
#  6. 主流程 (Main)
# =============================================
def main():
    print("\n" + "=" * 60)
    print(f" 🔥 STARTING ABLATION EXPERIMENT: STEP {ABLATION_STEP} 🔥")
    print("=" * 60)

    # -------------------------------------
    # Phase 1: CDAE Pretraining (预训练编码器)
    # -------------------------------------
    print("\n[Phase 1] Pretraining Autoencoder...")
    # 为了测试效率，可以在实际运行消融实验时将 N_SIM 改小（比如 10000），节省每次测试的时间
    clean_sim, noisy_sim = generate_simulated_batch(10000, duration=25, fs=32)
    clean_val, noisy_val = generate_simulated_batch(1000, duration=25, fs=32)

    with strategy.scope():
        cdae = define_cdae()

    cdae.fit(
        noisy_sim, clean_sim,
        validation_data=(noisy_val, clean_val),
        epochs=15,  # 消融测试中可适当减小
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
        verbose=1
    )

    pretrained_weights = {}
    for name in ['enc_conv1', 'enc_bn1', 'enc_conv2', 'enc_bn2', 'enc_conv3', 'enc_bn3']:
        pretrained_weights[name] = cdae.get_layer(name).get_weights()

    # 释放内存
    del cdae, clean_sim, noisy_sim, clean_val, noisy_val
    K.clear_session()
    gc.collect()

    # -------------------------------------
    # Phase 2: Fine-tuning (构建对应阶段网络并微调)
    # -------------------------------------
    print(f"\n[Phase 2] Fine-tuning DeepBeat Ablation Step {ABLATION_STEP} on Real Data...")

    if not os.path.exists(EXTRACTED_DIR):
        raise FileNotFoundError(f"Cannot find directory: {EXTRACTED_DIR}")

    # 内存映射读取大文件
    signals = np.load(os.path.join(EXTRACTED_DIR, 'signal.npy'), mmap_mode='r')
    qa_labels = np.load(os.path.join(EXTRACTED_DIR, 'qa_label.npy'), mmap_mode='r')
    rhythm_labels = np.load(os.path.join(EXTRACTED_DIR, 'rhythm.npy'), mmap_mode='r')
    p_data = np.load(os.path.join(EXTRACTED_DIR, 'parameters.npy'), allow_pickle=True)
    parameters = pd.DataFrame(p_data, columns=['timestamp', 'stream', 'ID'])

    # 按病人(ID)进行分割
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(np.arange(len(parameters)), groups=parameters['ID']))

    # 设置类别平衡权重
    weight_maps = {'qa': {0: 1.0, 1: 1.0, 2: 2.0}, 'rhythm': {0: 3.0, 1: 1.0}}

    train_gen = DataGenerator(signals, qa_labels, rhythm_labels, train_idx,
                              batch_size=BATCH_SIZE, shuffle=True, class_weights=weight_maps)
    val_gen = DataGenerator(signals, qa_labels, rhythm_labels, val_idx,
                            batch_size=BATCH_SIZE, shuffle=False)

    with strategy.scope():
        # ✨ 这里会根据顶部的 ABLATION_STEP 变量生成对应网络
        model = build_ablation_model(step=ABLATION_STEP, pretrained_weights=pretrained_weights)

        model.compile(
            optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
            loss={'qa': categorical_focal_loss(2.0, 0.25),
                  'rhythm': categorical_focal_loss(2.0, 0.25)},
            loss_weights={'qa': 1.0, 'rhythm': 1.0},
            metrics={'qa': 'accuracy', 'rhythm': 'accuracy'}
        )

    # 设置回调
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    ]

    print("\nStarting Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks,
        verbose=1,
        workers=8,
        use_multiprocessing=False  # Windows 必须 False
    )

    # 强制保存最优权重（已被 EarlyStopping 恢复）
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Training Complete. Best model for Step {ABLATION_STEP} saved as {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()