import numpy as np
from scipy.signal import detrend, resample
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, LeakyReLU, \
    BatchNormalization, Dropout, Activation, SeparableConv1D, GlobalAveragePooling1D, Reshape, Multiply, Bidirectional, \
    GRU, UpSampling1D  # 增加 UpSampling1D
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
# 0 = Baseline (原始版本，参数量最大)
# 1 = Baseline + 共享层轻量化 (SeparableConv & 通道减半)
# 2 = 阶段 1 + 分类器轻量化 (Dense 节点减半)
# 3 = 阶段 2 + 移除 BiGRU (最终 Lite 版本)
# =====================================================================
ABLATION_STEP = 3


MODEL_SAVE_PATH = f"deepbeat_ablation_step_{ABLATION_STEP}_no_pretrain_new.h5"

# =============================================
#  1. 全局与 GPU 配置 (与消融实验对齐)
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
EPOCHS_FINETUNE = 30
INITIAL_LR = 1e-3


# =============================================
#  2. 数据生成器 (与消融实验对齐，支持样本权重)
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

        # 归一化
        X = detrend(X, type='linear', axis=1)
        mean_val = np.mean(X, axis=1, keepdims=True)
        std_val = np.std(X, axis=1, keepdims=True)
        std_val[std_val == 0] = 1.0
        X = (X - mean_val) / std_val
        if X.ndim == 2:
            X = X[..., np.newaxis]

        y_qa = self.qa_labels[batch_indices].astype(np.float32)
        y_rhythm = self.rhythm_labels[batch_indices].astype(np.float32)

        # 样本权重计算 (与消融实验对齐)
        sample_weights = None
        if self.class_weights:
            r_classes = np.argmax(y_rhythm, axis=1)
            r_sw = np.array([self.class_weights['rhythm'].get(c, 1.0) for c in r_classes])
            qa_classes = np.argmax(y_qa, axis=1)
            qa_sw = np.array([self.class_weights['qa'].get(c, 1.0) for c in qa_classes])
            sample_weights = {'qa': qa_sw, 'rhythm': r_sw}

        return X, {'qa': y_qa, 'rhythm': y_rhythm}, sample_weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================
#  3. 损失函数 (Focal Loss，与消融实验完全一致)
# =============================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=-1)

    return focal_loss


# =============================================
#  [新增] 模拟信号生成 (用于 Phase 1 预训练)
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
#  4. 模型架构定义
# =============================================
def se_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    se = Reshape((1, filters))(se)
    return Multiply()([input_tensor, se])


def get_encoder_layers(input_tensor):
    """基础特征提取层"""
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


# [新增] 自编码器定义，用于 Phase 1
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


def build_ablation_model(step, pretrained_weights=None):  # [修改] 增加预训练权重接收参数
    """
    根据消融阶段构建模型（递进式轻量化设计）
    Step 0 = Baseline (标准卷积 + BiGRU + 全通道)
    Step 1 = 替换 SeparableConv (保持全通道 + 保持 BiGRU)
    Step 2 = Step 1 + 移除 BiGRU (替换为 GAP)
    Step 3 = Step 2 + 通道及节点数减半 (最终 Lite 版本)
    """
    input_sig = Input(shape=(INPUT_LENGTH, 1), name='input_signal')
    encoded = get_encoder_layers(input_sig)

    L2_REG = 1e-4

    # ==========================================
    # A: 共享层 (Shared Layers) 逻辑分配
    # ==========================================
    # 控制通道数: Step 3 时通道数减半
    shared_filters = [64, 128, 256] if step >= 3 else [128, 256, 512]

    if step >= 1:
        # Step 1, 2, 3: 使用深度可分离卷积 (SeparableConv1D)
        x = SeparableConv1D(shared_filters[0], 5, padding='same', name='shared_sepconv4')(encoded)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = se_block(x)
        x = Dropout(0.25)(x)

        x = SeparableConv1D(shared_filters[1], 5, padding='same', name='shared_sepconv5')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = se_block(x)
        x = Dropout(0.25)(x)

        x = SeparableConv1D(shared_filters[2], 5, padding='same', name='shared_sepconv6')(x)
        x = BatchNormalization()(x)
        shared_features = LeakyReLU(alpha=0.1)(x)
        shared_features = Dropout(0.25)(shared_features)
    else:
        # Step 0 (Baseline): 使用普通卷积 (Conv1D)
        x = Conv1D(shared_filters[0], 5, padding='same', kernel_initializer='he_normal', name='shared_conv4')(encoded)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = se_block(x)
        x = Dropout(0.25)(x)

        x = Conv1D(shared_filters[1], 5, padding='same', kernel_initializer='he_normal', name='shared_conv5')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = se_block(x)
        x = Dropout(0.25)(x)

        x = Conv1D(shared_filters[2], 5, padding='same', kernel_initializer='he_normal', name='shared_conv6')(x)
        x = BatchNormalization()(x)
        shared_features = LeakyReLU(alpha=0.1)(x)
        shared_features = Dropout(0.25)(shared_features)

    # ==========================================
    # B: 分类器全连接层 (Dense) 维度分配
    # ==========================================
    # 控制节点数: Step 3 时全连接层节点也随之减半
    dense_dim_1 = 64 if step >= 3 else 128
    dense_dim_2 = 32 if step >= 3 else 64
    drop_rate = 0.5

    # --- QA 分支 ---
    qa_x = GlobalAveragePooling1D()(shared_features)
    qa_x = Dense(dense_dim_1, activation='relu')(qa_x)
    qa_x = Dropout(drop_rate)(qa_x)
    qa_x = Dense(dense_dim_2, activation='relu')(qa_x)
    qa_out = Dense(3, activation='softmax', name='qa')(qa_x)

    # ==========================================
    # C: Rhythm 分支逻辑分配
    # ==========================================
    rhythm_filters = 128 if step >= 3 else 256

    # 1. 卷积类型替换
    if step >= 1:
        r_x = SeparableConv1D(rhythm_filters, 3, padding='same', activation='relu')(shared_features)
    else:
        r_x = Conv1D(rhythm_filters, 3, padding='same', activation='relu')(shared_features)

    r_x = BatchNormalization()(r_x)
    r_x = MaxPooling1D(2)(r_x)

    # 2. 序列建模层替换 (核心消融点: BiGRU vs GAP)
    if step >= 2:
        # Step 2, 3: 移除 BiGRU，使用 GAP 降低参数量并防止过拟合
        r_x = GlobalAveragePooling1D()(r_x)
    else:
        # Step 0, 1: 保留 BiGRU 捕捉时序特征
        r_x = Bidirectional(GRU(64, return_sequences=False))(r_x)

    # 3. 最终分类输出
    r_x = Dense(dense_dim_1, activation='relu')(r_x)
    r_x = Dropout(drop_rate)(r_x)
    r_x = Dense(dense_dim_2, activation='relu')(r_x)
    rhythm_out = Dense(2, activation='softmax', name='rhythm')(r_x)

    model = Model(inputs=input_sig, outputs=[qa_out, rhythm_out], name=f'DeepBeat_Step_{step}')

    # 载入 Phase 1 预训练权重
    if pretrained_weights:
        print("Transferring pretrained CDAE encoder weights...")
        for name in ['enc_conv1', 'enc_bn1', 'enc_conv2', 'enc_bn2', 'enc_conv3', 'enc_bn3']:
            try:
                model.get_layer(name).set_weights(pretrained_weights[name])
            except Exception as e:
                print(f" ! Failed to load {name}: {e}")

    model.summary()
    return model


# =============================================
#  5. 主流程 (与消融实验对齐)
# =============================================
def main():
    print(f"\n🚀 STARTING TRAINING: STEP {ABLATION_STEP} (With Pretraining)")
    print(f"   Learning rate: {INITIAL_LR}")
    print(f"   Epochs: {EPOCHS_FINETUNE}")
    print(f"   Loss: Focal Loss (gamma=2.0, alpha=0.25)")
    print("=" * 40)

    # -------------------------------------
    # [新增] Phase 1: CDAE Pretraining (预训练编码器)
    # -------------------------------------
    print("\n[Phase 1] Pretraining Autoencoder...")
    clean_sim, noisy_sim = generate_simulated_batch(10000, duration=25, fs=32)
    clean_val, noisy_val = generate_simulated_batch(1000, duration=25, fs=32)

    with strategy.scope():
        cdae = define_cdae()

    cdae.fit(
        noisy_sim, clean_sim,
        validation_data=(noisy_val, clean_val),
        epochs=15,
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
    # Phase 2: Fine-tuning on Real Data
    # -------------------------------------
    print(f"\n[Phase 2] Fine-tuning DeepBeat Ablation Step {ABLATION_STEP} on Real Data...")

    if not os.path.exists(EXTRACTED_DIR):
        raise FileNotFoundError(f"Cannot find: {EXTRACTED_DIR}")

    # 加载真实数据
    print("Loading real dataset from .npy files...")
    signals = np.load(os.path.join(EXTRACTED_DIR, 'signal.npy'), mmap_mode='r')
    qa_labels = np.load(os.path.join(EXTRACTED_DIR, 'qa_label.npy'), mmap_mode='r')
    rhythm_labels = np.load(os.path.join(EXTRACTED_DIR, 'rhythm.npy'), mmap_mode='r')
    p_data = np.load(os.path.join(EXTRACTED_DIR, 'parameters.npy'), allow_pickle=True)
    parameters = pd.DataFrame(p_data, columns=['timestamp', 'stream', 'ID'])

    all_indices = np.arange(len(parameters))

    # 按病人划分
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(all_indices, groups=parameters['ID']))

    # 样本权重配置
    qa_cw_dict = {0: 1.0, 1: 1.0, 2: 2.0}
    r_cw_dict = {0: 3.0, 1: 1.0}
    weight_maps = {'qa': qa_cw_dict, 'rhythm': r_cw_dict}

    train_gen = DataGenerator(signals, qa_labels, rhythm_labels, train_idx, batch_size=BATCH_SIZE, shuffle=True,
                              class_weights=weight_maps)
    val_gen = DataGenerator(signals, qa_labels, rhythm_labels, val_idx, batch_size=BATCH_SIZE, shuffle=False,
                            class_weights=None)

    with strategy.scope():
        # 传递提取出的预训练权重
        model = build_ablation_model(step=ABLATION_STEP, pretrained_weights=pretrained_weights)

        # 编译配置与消融实验对齐
        loss_fn = categorical_focal_loss(gamma=2.0, alpha=0.25)
        model.compile(
            optimizer=Adam(learning_rate=INITIAL_LR, clipnorm=1.0),
            loss={'qa': loss_fn, 'rhythm': loss_fn},
            loss_weights={'qa': 1.0, 'rhythm': 1.0},
            metrics={'qa': 'accuracy', 'rhythm': 'accuracy'}
        )

    # Callbacks 配置与消融实验对齐
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)  # patience=8 与消融实验对齐
    ]

    print("\nStarting Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks,
        verbose=1,
        workers=8,
        use_multiprocessing=False
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Training completed. Best Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()