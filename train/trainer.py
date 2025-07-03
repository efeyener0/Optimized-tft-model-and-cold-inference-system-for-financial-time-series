
import os
import sys
import yaml
import shutil
import argparse
from typing import Dict, Set
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# DOĞRU İMPORT: Kütüphanenin en güncel halinde, ana sınıf compile-dostu hale getirildi.
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer, EncoderNormalizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from torch.utils.data import TensorDataset, DataLoader, Subset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# --- PERFORMANS AYARLARI ---
is_tf32_enabled = False
if torch.cuda.is_available():
    if torch.cuda.get_device_capability()[0] >= 8:
        print("[PERFORMANCE] Ampere veya daha yeni GPU algılandı. TF32 etkinleştiriliyor.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        is_tf32_enabled = True

    import torch._dynamo
    torch._dynamo.config.capture_scalar_outputs = True
    print("[PERFORMANCE] torch.compile için skaler çıktı yakalama etkinleştirildi.")

# --- YARDIMCI CALLBACK'LER ---
class BestModelToPthCallback(pl.Callback):
    """
    Validasyon setinde en iyi skoru elde eden modelin checkpoint'ini
    sadece state_dict'i içerecek şekilde ayrı bir .pth dosyası olarak kaydeder.
    """
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        self.last_best_path = None
        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.is_global_zero: return
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path) and best_model_path != self.last_best_path:
            self.last_best_path = best_model_path
            target_pth_path = os.path.join(self.save_dir, "best_model_weights.pth")
            try:
                checkpoint = torch.load(best_model_path, map_location="cpu")
                state_dict = checkpoint['state_dict']
                torch.save(state_dict, target_pth_path)
                pl_module.print(f"\n[CALLBACK] Yeni en iyi model bulundu! Ağırlıklar kaydedildi:\n"
                              f"           -> {target_pth_path}\n"
                              f"           -> Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint_callback.best_model_score:.6f}")
            except Exception as e:
                pl_module.print(f"\n[CALLBACK HATA] En iyi modelin .pth kopyası oluşturulamadı: {e}")

class InferenceConfigCallback(pl.Callback):
    """
    Eğitim tamamlandığında, inference için gerekli olan normalizasyon metriklerini
    ve model yapılandırmasını bir YAML dosyasına kaydeder.
    """
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.is_global_zero:
            return

        pl_module.print("\n[CALLBACK] Eğitim tamamlandı. Inference için yapılandırma dosyası oluşturuluyor...")

        datamodule = trainer.datamodule
        if not hasattr(datamodule, 'training_ts_dataset'):
            pl_module.print("[HATA] DataModule'de 'training_ts_dataset' bulunamadı.")
            return

        def to_human_readable(value):
            if isinstance(value, (np.ndarray, torch.Tensor)):
                value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                return value.item() if value.size == 1 else value.tolist()
            if isinstance(value, list) and len(value) == 1:
                return value[0]
            return value

        target_scaler_info = {}
        if hasattr(datamodule, 'target_mean') and datamodule.target_mean is not None and \
           hasattr(datamodule, 'target_std') and datamodule.target_std is not None:
            params = {
                'mean': to_human_readable(datamodule.target_mean),
                'std': to_human_readable(datamodule.target_std)
            }
            target_scaler_info = {
                'target_column': datamodule.hparams.model['target'],
                'type': 'StandardScaler',
                'params': params
            }
        else:
             pl_module.print("[UYARI] Hedef değişken için scaler metrikleri DataModule'de bulunamadı.")

        feature_scalers_info = {}
        dataset_params = datamodule.training_ts_dataset.get_parameters()
        other_encoders = {}
        if dataset_params.get('scalers'):
            other_encoders.update(dataset_params['scalers'])
        if dataset_params.get('categorical_encoders'):
            other_encoders.update(dataset_params['categorical_encoders'])

        for key, encoder in other_encoders.items():
            scaler = encoder
            scaler_type = scaler.__class__.__name__
            params = {}

            if hasattr(scaler, 'center_') and hasattr(scaler, 'scale_'):
                params['mean'] = to_human_readable(scaler.center_)
                params['std'] = to_human_readable(scaler.scale_)
            elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                params['mean'] = to_human_readable(scaler.mean_)
                params['std'] = to_human_readable(scaler.scale_)
            elif isinstance(scaler, NaNLabelEncoder) and hasattr(scaler, 'classes_'):
                params['classes'] = {str(k): int(v) for k, v in scaler.classes_.items()}

            if params:
                feature_scalers_info[key] = {'type': scaler_type, 'params': params}

        model_config = trainer.datamodule.hparams.get('model', {})
        inference_config = {
            'model_definition': {
                'target': model_config.get('target'),
                'group_ids': model_config.get('group_ids', []),
                'sequence_length': model_config.get('sequence_length'),
                'prediction_steps': model_config.get('prediction_steps'),
                'static_categoricals': model_config.get('static_categoricals', []),
                'static_reals': model_config.get('static_reals', []),
                'time_varying_known_categoricals': model_config.get('time_varying_known_categoricals', []),
                'time_varying_known_reals': model_config.get('time_varying_known_reals', []),
                'time_varying_unknown_categoricals': model_config.get('time_varying_unknown_categoricals', []),
                'time_varying_unknown_reals': model_config.get('time_varying_unknown_reals', []),
            },
            'target_scaler': target_scaler_info,
            'feature_scalers': feature_scalers_info,
            'notes': {
                'description': 'Bu dosya, eğitimli modelle inference yapmak için gereken '
                               'yapılandırma ve normalizasyon/kodlama metriklerini içerir.',
                'creation_timestamp_utc': datetime.utcnow().isoformat()
            }
        }

        output_path = os.path.join(self.save_dir, "inference_params.yaml")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(inference_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            pl_module.print(f"[CALLBACK] Inference yapılandırması başarıyla kaydedildi:\n           -> {output_path}")
        except Exception as e:
            pl_module.print(f"\n[CALLBACK HATA] Inference yapılandırma dosyası kaydedilemedi: {e}")


# --- VERİ MODÜLÜ ---
class GpuPreloadedDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.training_ts_dataset = None
        self.full_train_gpu_dataset = None
        self.full_val_gpu_dataset = None
        self.x_keys = None
        self.dynamic_stride_range = self.hparams.training.get('dynamic_stride_range', [1, 1])
        self.target_mean = None
        self.target_std = None


    def _validate_dataframe(self, df: pd.DataFrame, config: Dict):
        model_cfg = config['model']
        required_cols: Set[str] = {"time_idx", model_cfg['target']}
        required_cols.update(model_cfg.get('group_ids', []))
        required_cols.update(model_cfg.get('time_varying_known_reals', []))
        required_cols.update(model_cfg.get('time_varying_unknown_reals', []))
        if missing_cols := required_cols - set(df.columns):
            raise ValueError(f"Veri dosyasında eksik sütunlar var: {sorted(list(missing_cols))}")

    def setup(self, stage: str = None):
        if self.full_train_gpu_dataset is not None: return
        cfg = self.hparams

        try:
            # Veri yolu config dosyasından okunuyor, artık hata vermemeli.
            data = pd.read_csv(cfg['data']['path'], encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Veri dosyası bulunamadı! Yol: '{cfg['data']['path']}'.")

        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values('datetime').reset_index(drop=True)
        data["time_idx"] = data.index

        for group_id in cfg['model'].get('group_ids', ['group']):
            if group_id not in data.columns: data[group_id] = "default_group"
        self._validate_dataframe(data, cfg)

        print("\n[Veri Bölümleme] Zaman serisi bütünlüğünü korumak için zaman-tabanlı bölme uygulanıyor.")
        training_cutoff_idx = int(len(data) * 0.8)
        train_df = data.iloc[:training_cutoff_idx]
        val_df = data.iloc[training_cutoff_idx - cfg['model']['sequence_length']:]

        self.training_ts_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=cfg['model']['target'],
            group_ids=cfg['model']['group_ids'],
            max_encoder_length=cfg['model']['sequence_length'],
            max_prediction_length=cfg['model']['prediction_steps'],
            time_varying_known_reals=cfg['model'].get('time_varying_known_reals', []),
            time_varying_unknown_reals=cfg['model'].get('time_varying_unknown_reals', []),
            add_relative_time_idx=True,
            allow_missing_timesteps=True,
        )

        normalizer = self.training_ts_dataset.get_parameters()["target_normalizer"]

        if hasattr(normalizer, 'center_') and hasattr(normalizer, 'scale_'):
            self.target_mean = torch.tensor(normalizer.center_, dtype=torch.float32)
            self.target_std = torch.tensor(normalizer.scale_, dtype=torch.float32)
        else:
            raise AttributeError(
                f"Hedef normalizer'dan (tip: {type(normalizer).__name__}) beklenen 'center_' ve 'scale_' "
                f"özellikleri bulunamadı. Lütfen normalizer'ın yapısını kontrol edin. "
                f"Normalizer __dict__: {getattr(normalizer, '__dict__', {})}"
            )

        print(f"[METRİK ÇIKARIMI] '{cfg['model']['target']}' için parametreler başarıyla çekildi: Mean={self.target_mean.item():.6f}, Std={self.target_std.item():.6f}")

        validation_ts_dataset = TimeSeriesDataSet.from_dataset(self.training_ts_dataset, val_df, stop_randomization=True)

        print("[Veri Dönüşümü] Veri seti, DataLoader aracılığıyla tensörlere dönüştürülüyor...")
        train_loader = self.training_ts_dataset.to_dataloader(train=True, batch_size=len(self.training_ts_dataset), num_workers=0, shuffle=False)
        val_loader = validation_ts_dataset.to_dataloader(train=False, batch_size=len(validation_ts_dataset), num_workers=0, shuffle=False)
        train_x_cpu, (train_y_cpu, _) = next(iter(train_loader))
        val_x_cpu, (val_y_cpu, _) = next(iter(val_loader))

        self.x_keys = sorted(list(train_x_cpu.keys()))
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_x_gpu = {k: v.to(target_device) for k, v in train_x_cpu.items()}
        val_x_gpu = {k: v.to(target_device) for k, v in val_x_cpu.items()}
        train_y_gpu = train_y_cpu.to(target_device)
        val_y_gpu = val_y_cpu.to(target_device)

        train_tensors_for_dataset = [train_x_gpu[key] for key in self.x_keys]
        val_tensors_for_dataset = [val_x_gpu[key] for key in self.x_keys]

        self.full_train_gpu_dataset = TensorDataset(*train_tensors_for_dataset, train_y_gpu)
        self.full_val_gpu_dataset = TensorDataset(*val_tensors_for_dataset, val_y_gpu)

        print(f"Kurulum Tamamlandı: {len(self.full_train_gpu_dataset)} eğitim ve {len(self.full_val_gpu_dataset)} validasyon örneği GPU'da hazır.")

    def _generate_dynamic_stride_indices(self) -> list:
        total_samples = len(self.full_train_gpu_dataset)
        min_stride, max_stride = self.dynamic_stride_range
        print(f"\n[Dynamic Stride] Yeni epoch için indeksler oluşturuluyor. Aralık: [{min_stride}, {max_stride}]")

        estimated_steps = total_samples
        strides = torch.randint(low=min_stride, high=max_stride + 1, size=(estimated_steps,), dtype=torch.long)
        indices = torch.cumsum(strides, dim=0)
        indices = torch.cat((torch.tensor([0], dtype=torch.long), indices[:-1]))
        valid_indices = indices[indices < total_samples]

        print(f"[Dynamic Stride] Orijinal boyut: {total_samples}, Yeni boyut: {len(valid_indices)}")
        return valid_indices.tolist()

    def train_dataloader(self):
        dataset_to_use = self.full_train_gpu_dataset
        min_stride, max_stride = self.dynamic_stride_range
        if not (min_stride == 1 and max_stride == 1):
            indices = self._generate_dynamic_stride_indices()
            dataset_to_use = Subset(self.full_train_gpu_dataset, indices)
        return DataLoader(dataset_to_use, batch_size=self.hparams.training['batch_size'], shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.full_val_gpu_dataset, batch_size=self.hparams.training['batch_size'] * 2, shuffle=False, num_workers=0, pin_memory=False)


# --- KAYIP FONKSİYONLARI ---
class _Db4Filters(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        # DÜZELTME: Gerçek db4 (Daubechies-4, 8 katsayı) filtreleri kullanılıyor.
        l = torch.tensor([
            0.23037781, 0.71484657, 0.63088077, -0.02798377,
           -0.18703481, 0.03084138, 0.03288301, -0.0105974
        ])
        h = torch.tensor([
           -0.0105974, -0.03288301, 0.03084138, 0.18703481,
           -0.02798377, -0.63088077, 0.71484657, -0.23037781
        ])
        self.register_buffer('low_pass_filter', l.flip(0).view(1, 1, -1).repeat(c, 1, 1))
        self.register_buffer('high_pass_filter', h.flip(0).view(1, 1, -1).repeat(c, 1, 1))
        self.num_channels, self.filter_len = c, l.numel()

    def forward(self, x, dilation):
        low_pass = self.low_pass_filter.to(x.device, x.dtype)
        high_pass = self.high_pass_filter.to(x.device, x.dtype)
        padding = (self.filter_len - 1) * dilation
        x_padded = F.pad(x, (padding // 2, padding - padding // 2), 'reflect')
        return F.conv1d(x_padded, low_pass, groups=self.num_channels, dilation=dilation), \
               F.conv1d(x_padded, high_pass, groups=self.num_channels, dilation=dilation)

class HybridFinancialLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        [setattr(self, k, v) for k, v in kwargs.items()]
        self.filters = _Db4Filters(c=1)

    def _metric(self, error, metric_name):
        if metric_name == 'rmse': return torch.sqrt(error.pow(2).mean() + self.eps)
        if metric_name == 'mae': return error.abs().mean()
        return error.pow(2).mean() # mse

    def _wavelet_term(self, y_hat, y):
        if y_hat.size(-1) <= 1:
            return torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)

        y_hat_approx, y_approx = y_hat, y
        wavelet_loss = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)

        for i in range(self.levels):
            if y_hat_approx.size(-1) < 2**(i+1):
                break
            y_hat_approx, y_hat_detail = self.filters(y_hat_approx, 2**i)
            y_approx, y_detail = self.filters(y_approx, 2**i)
            wavelet_loss += self._metric(y_hat_approx - y_approx, self.wavelet_metric)
            wavelet_loss += self._metric(y_hat_detail - y_detail, self.wavelet_metric)

        return wavelet_loss / (2 * self.levels) if self.levels > 0 else torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)

    def forward(self, y_hat, y):
        point_loss = self._metric(y_hat - y, self.point_metric)

        if y_hat.size(-1) > 1:
            diff_loss = self._metric(torch.diff(y_hat, dim=-1) - torch.diff(y, dim=-1), self.diff_metric)
        else:
            diff_loss = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)

        wavelet_loss = self._wavelet_term(y_hat, y)
        total_loss = self.gamma * point_loss + self.beta * diff_loss + self.alpha * wavelet_loss
        return {"total": total_loss, "point": point_loss, "diff": diff_loss, "wavelet": wavelet_loss}


# --- ANA LIGHTNING MODELİ ---
class LightningTFT(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = HybridFinancialLoss(**self.hparams.loss_params)
        self.model = None

    def _custom_weight_init(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.uniform_(m.weight, a=self.hparams.training['custom_init_range'][0], b=self.hparams.training['custom_init_range'][1])
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def setup(self, stage: str):
        if self.model is not None: return
        if stage != 'fit': return

        dataset = self.trainer.datamodule.training_ts_dataset

        # DÜZELTME: Güncel kütüphanede, ana sınıf zaten compile-dostu.
        print("\n[Model Kurulumu] Güncellenmiş (compile-dostu) ana TemporalFusionTransformer modülü kullanılıyor.")
        model_to_process = TemporalFusionTransformer.from_dataset(
            dataset, **self.hparams.model_params, loss=MAE()
        )

        if self.hparams.training.get('use_custom_weight_init', False):
            print("\n[Ağırlık Başlangıcı] Özel, seed tabanlı yöntem uygulanıyor.")
            init_seed = self.hparams.training.get('custom_init_seed', 42)
            torch.manual_seed(init_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(init_seed)
            print(f"  -> Başlangıç Seed'i: {init_seed}, Aralığı: {self.hparams.training['custom_init_range']}")
            model_to_process.apply(self._custom_weight_init)
            pl.seed_everything(self.hparams.training['seed'], workers=True)

        model_on_correct_device = model_to_process.to(self.device)
        print(f"\n[Model Kurulumu] TFT modeli '{str(self.device)}' cihazına taşındı.")

        if self.hparams.training.get('compile_model', False):
            print("[Optimizasyon] torch.compile() etkinleştiriliyor (mod: reduce-overhead)...")
            self.model = torch.compile(model_on_correct_device, mode="reduce-overhead")
        else:
            self.model = model_on_correct_device

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_on_device = {k: v.to(self.device) for k, v in x.items()}
        return self.model(x_on_device)

    def _shared_step(self, batch, stage: str):
        target_device = self.device
        x_tensors_tuple = tuple(t.to(target_device) for t in batch[:-1])
        y_actual_fp32 = batch[-1].to(target_device)
        x_dict = {key: tensor for key, tensor in zip(self.trainer.datamodule.x_keys, x_tensors_tuple)}

        predictions_fp32 = self(x_dict)['prediction']
        if predictions_fp32.dim() == 3 and predictions_fp32.shape[-1] == 1:
            predictions_fp32 = predictions_fp32.squeeze(-1)

        predictions_fp64 = predictions_fp32.to(torch.float64)
        y_actual_fp64 = y_actual_fp32.to(torch.float64)

        if predictions_fp64.shape != y_actual_fp64.shape:
            y_actual_fp64 = y_actual_fp64[..., :predictions_fp64.shape[-1]]

        loss_dict = self.loss_fn(predictions_fp64.unsqueeze(1), y_actual_fp64.unsqueeze(1))
        total_loss = loss_dict["total"].to(torch.float32)

        self.log(f"{stage}/loss", total_loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log_dict({f"loss/{stage}_{k}": v.to(torch.float32) for k, v in loss_dict.items() if k != 'total'},
                      on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx): return self._shared_step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer_params
        sched_params = self.hparams.lr_scheduler_params
        optimizer = torch.optim.Adam(self.parameters(), **opt_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_params.get('params', {}))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": sched_params.get('monitor', 'val/loss')}}


# --- EĞİTİM YÖNETİMİ ---
def validate_config(config: Dict):
    required_keys = ['project_name', 'run_name', 'data', 'model', 'training', 'logging', 'trainer_params']
    if missing_keys := [k for k in required_keys if k not in config]:
        raise KeyError(f"Config hatası: Eksik anahtarlar: {missing_keys}")

    required_model_keys = ['target', 'group_ids', 'sequence_length', 'prediction_steps']
    if missing_model_keys := [k for k in required_model_keys if k not in config['model']]:
        raise KeyError(f"Config `model` bölümünde eksik anahtarlar: {missing_model_keys}")

def train(config: Dict):
    pl.seed_everything(config['training']['seed'], workers=True)

    log_dir = os.path.join(config['logging']['log_dir'], config['project_name'])
    run_dir = os.path.join(log_dir, config['run_name'])
    ckpt_dir = os.path.join(run_dir, 'checkpoints')

    datamodule = GpuPreloadedDataModule(config)
    model = LightningTFT(**config)
    logger = TensorBoardLogger(save_dir=log_dir, name=config['run_name'], version=0)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename='{epoch:02d}-{val/loss:.6f}',
        monitor="val/loss", mode="min",
        save_top_k=config['logging']['save_top_k'], save_last=True
    )

    callbacks = [
        checkpoint_cb,
        EarlyStopping(monitor="val/loss", patience=10, mode="min", verbose=True),
        LearningRateMonitor(logging_interval='epoch'),
        BestModelToPthCallback(save_dir=run_dir),
        InferenceConfigCallback(save_dir=run_dir)
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        **config['trainer_params']
    )

    print("\n" + "=" * 60 + "\n HİBRİT HASSASİYET (FP64/TF32/FP32) EĞİTİM MODU \n" + "=" * 60)
    print(f"- Temel Hassasiyet: {trainer.precision_plugin.precision}")
    print(f"- Matris İşlemleri: {'TF32 (Tensor Core)' if is_tf32_enabled else 'FP32 (CUDA Core)'}")
    print(f"- Kayıp Fonksiyonu: FP64 (Maksimum Stabilite)")
    print(f"- Torch Compile: {'ETKİN' if config['training'].get('compile_model', False) else 'DEVRE DIŞI'}")
    print("=" * 60 + "\n")

    ckpt_path = os.path.join(ckpt_dir, "last.ckpt") if config['training']['mode'] == 'resume' and os.path.exists(os.path.join(ckpt_dir, "last.ckpt")) else None
    if ckpt_path: print(f"Eğitime şuradan devam ediliyor: {ckpt_path}")

    print(f"Eğitim başlıyor: Proje={config['project_name']}, Deney={config['run_name']}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print("\nEğitim tamamlandı.")

    final_best_path = os.path.join(run_dir, "best_model_weights.pth")
    if os.path.exists(final_best_path):
        print(f"Inference için en iyi modelin son hali:\n -> {final_best_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Üretime Hazır, Hibrit Hassasiyetli ve Optimize Edilmiş TFT Eğitim Scripti.")
    parser.add_argument('--config', type=str, required=True, help='Yapılandırma dosyasının yolu (zorunlu).')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"HATA: Konfigürasyon dosyası bulunamadı: '{args.config}'", file=sys.stderr)
        sys.exit(1)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    try:
        validate_config(config)
        train(config)
    except Exception as e:
        print(f"\n!!! BEKLENMEDİK BİR HATA OLUŞTU !!!\n Tip: {type(e).__name__}\n Mesaj: {e}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
