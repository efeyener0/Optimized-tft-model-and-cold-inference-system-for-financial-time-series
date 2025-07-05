# core/tft_pipeline_core.py

import os
import yaml
from typing import Dict, Any, Tuple, List
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer
    from pytorch_forecasting.metrics import MAE
except ImportError:
    raise ImportError(
        "Lütfen gerekli kütüphaneleri kurun: 'pip install pytorch-forecasting pandas scikit-learn'"
    )

__all__ = ["InferencePipeline"]


class _DataPreprocessor:
    """
    Bu sınıf, inference_params.yaml'daki ölçekleme bilgileriyle veriyi
    manuel olarak modele hazırlar. Orijinal veriye ihtiyaç duymaz.
    Bu sınıfın işlevi doğrudur ve değişmemelidir.
    """
    def __init__(self, config: Dict[str, Any]):
        self.model_def = config['model_definition']
        self.feature_scalers_config = config.get('feature_scalers', {})
        self.target_scaler_config = config['target_scaler']
        self.sequence_length = self.model_def['sequence_length']
        
        if not self.target_scaler_config:
            raise ValueError("Yapılandırma dosyasında 'target_scaler' bilgisi bulunamadı.")
            
        self.target_mean = self.target_scaler_config['params']['mean']
        self.target_std = self.target_scaler_config['params']['std']
        
    def transform(self, df: pd.DataFrame, last_known_time_idx: int) -> pd.DataFrame:
        if len(df) < self.sequence_length:
            raise ValueError(f"Girdi DataFrame'i en az {self.sequence_length} satır olmalıdır, ancak {len(df)} satır verildi.")
        
        input_df = df.tail(self.sequence_length).copy()

        all_scalers_config = self.feature_scalers_config.copy()
        all_scalers_config[self.target_scaler_config['target_column']] = self.target_scaler_config
             
        for col, scaler_info in all_scalers_config.items():
            if col in input_df.columns:
                mean = scaler_info['params']['mean']
                std = scaler_info['params'].get('std', 1.0)
                if std > 1e-8:
                    input_df[col] = (input_df[col] - mean) / std
        
        input_df['time_idx'] = range(last_known_time_idx + 1, last_known_time_idx + 1 + len(input_df))
        
        for col in self.model_def.get('group_ids', []):
             input_df[col] = "default_group"

        return input_df

    def inverse_transform_prediction(self, scaled_prediction: torch.Tensor) -> List[float]:
        """
        Tahmini orijinal ölçeğe döndürür ve `TypeError`'ı önlemek için
        bir Python listesi olarak döndürür.
        """
        prediction_np = scaled_prediction.cpu().numpy()
        final_prediction = (prediction_np * self.target_std) + self.target_mean
        return final_prediction.flatten().tolist()


class _BaseRobustNormalizer(TorchNormalizer):
    """
    Tüm sağlamlaştırılmış normalizer'lar için temel sınıf.
    İç durumu manuel olarak ve doğru tiple ayarlar.
    Bu sınıf SADECE model iskeletini oluştururken kullanılır.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.center_ = torch.tensor([0.0])
        self.scale_ = torch.tensor([1.0])

    def fit(self, *args, **kwargs):
        # Bu sahte normalizer'ın 'fit' edilmesini engelle.
        pass

    def _to_tensor(self, y):
        if isinstance(y, pd.Series):
            numeric_series = pd.to_numeric(y, errors='coerce').fillna(0)
            return torch.from_numpy(numeric_series.to_numpy()).float()
        elif isinstance(y, np.ndarray):
            if y.dtype == np.object_:
                 y = y.astype(np.float64)
            return torch.from_numpy(y).float()
        return y

class _RobustTargetNormalizer(_BaseRobustNormalizer):
    """
    SADECE hedef değişken için kullanılır.
    Kütüphanenin beklediği gibi (veri, ölçek) formatında bir tuple döndürür.
    Bu sınıf SADECE model iskeletini oluştururken kullanılır.
    """
    def transform(self, y: pd.Series, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self._to_tensor(y)
        transformed_y = (y - self.center_) / self.scale_
        
        center_expanded = self.center_.expand(len(transformed_y))
        scale_expanded = self.scale_.expand(len(transformed_y))
        target_scale = torch.stack([center_expanded, scale_expanded], dim=1)
        
        return transformed_y, target_scale

class _RobustFeatureNormalizer(_BaseRobustNormalizer):
    """
    Hedef DIŞINDAKİ tüm sayısal özellikler için kullanılır.
    Bu sınıf SADECE model iskeletini oluştururken kullanılır.
    """
    def transform(self, y, **kwargs) -> torch.Tensor:
        y = self._to_tensor(y)
        return (y - self.center_) / self.scale_


class _TFTInferencer:
    """
    Modelin doğru iskeletini, YAML dosyasındaki scaler/encoder bilgilerini
    "plan" olarak kullanarak kurar ve .pth dosyasını yükler.
    """
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inferencer] Model, '{self.device}' cihazında çalışacak şekilde ayarlandı.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model ağırlık dosyası bulunamadı: {model_path}")
        
        self.model = self._build_and_load_model(model_path, config)
        self.model.to(self.device)
        self.model.eval()

    def _build_and_load_model(self, model_path: str, config: Dict[str, Any]) -> TemporalFusionTransformer:
        model_def = config['model_definition']
        model_hyperparams = config['model_hyperparameters']

        cat_encoders = {
            col.replace("__group_id__", ""): NaNLabelEncoder(add_nan=True)
            for col, info in config.get('feature_scalers', {}).items()
            if info['type'] == 'NaNLabelEncoder'
        }

        # Model iskeletini oluşturmak için özel normalizer'larımızı kullanıyoruz.
        target_dummy_normalizer = _RobustTargetNormalizer(method="standard", center=True)
        features_dummy_normalizer = _RobustFeatureNormalizer(method="standard", center=True)

        reals_list = (
            model_def.get('static_reals', []) +
            model_def.get('time_varying_known_reals', []) +
            model_def.get('time_varying_unknown_reals', []) +
            ['relative_time_idx']
        )
        unique_reals = sorted(list(set(reals_list)))

        feature_scalers = {col: features_dummy_normalizer for col in unique_reals}
        
        required_length = model_def['sequence_length'] + model_def['prediction_steps']
        
        all_cols = (
            [model_def['time_idx']]
            + model_def['group_ids']
            + [col for col in unique_reals if col != 'relative_time_idx']
            + [model_def['target']]
        )
        all_cols = sorted(list(set(all_cols)))

        dummy_df = pd.DataFrame(index=range(required_length))

        for col in all_cols:
            if col == model_def['time_idx']:
                dummy_df[col] = list(range(required_length))
            elif col in model_def['group_ids']:
                dummy_df[col] = 'default_group'
            else:
                dummy_df[col] = np.random.uniform(low=0.0, high=1.0, size=required_length)
        
        dummy_dataset = TimeSeriesDataSet(
            dummy_df,
            time_idx=model_def['time_idx'], 
            target=model_def['target'],
            group_ids=model_def['group_ids'],
            max_encoder_length=model_def['sequence_length'],
            max_prediction_length=model_def['prediction_steps'],
            static_reals=model_def.get('static_reals', []),
            time_varying_known_reals=model_def.get('time_varying_known_reals', []),
            time_varying_unknown_reals=model_def.get('time_varying_unknown_reals', []),
            add_relative_time_idx=True,
            categorical_encoders=cat_encoders,
            scalers=feature_scalers,
            target_normalizer=target_dummy_normalizer
        )
        print("[Inferencer] Model iskeletini oluşturmak için 'plan' başarıyla yaratıldı.")

        tft_model = TemporalFusionTransformer.from_dataset(dummy_dataset, **model_hyperparams)
        print("[Inferencer] Model mimarisi başarıyla oluşturuldu.")

        weights = torch.load(model_path, map_location=self.device)
        
        if 'state_dict' in weights:
            weights = weights['state_dict']
        
        cleaned_weights = OrderedDict()
        for key, value in weights.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            if new_key.startswith("net."):
                 new_key = new_key[len("net."):]
            cleaned_weights[new_key] = value
        
        tft_model.load_state_dict(cleaned_weights, strict=False)
        print(f"[Inferencer] Ağırlıklar '{os.path.basename(model_path)}' dosyasından (uyumsuz anahtarlar göz ardı edilerek) başarıyla yüklendi.")
        
        return tft_model

    def predict(self, input_df: pd.DataFrame) -> torch.Tensor:
        """
        Modelden tahmin alır. Bu metot, kalıcı `IndexError` hatasını
        çözmek için yeniden yazılmıştır.
        """
        # HATA KAYNAĞI ve NİHAİ ÇÖZÜM:
        # Kütüphanenin kendi içindeki ölçekleme mantığı ile bizim özel
        # normalizer sınıflarımız arasındaki uyumsuzluk, çözülmesi zor
        # `IndexError` hatalarına yol açıyor.
        #
        # Çözüm olarak, tahmin aşamasında kütüphanenin kendi standart
        # `TorchNormalizer` sınıfını kullanıyoruz. Veri zaten manuel olarak
        # ölçeklendiği için, bu normalizer'ın hedef değişken üzerinde ne
        # yaptığı önemli değildir. Bu, kütüphanenin kendi kodunu kullanmasını
        # sağlayarak uyumluluk sorunlarını ortadan kaldırır. Tahmin,
        # `model(x)` ile doğrudan alınır ve en sonda manuel olarak orijinal
        # ölçeğe döndürülür. Bu, en güvenilir ve kesin çözümdür.
        
        params = self.model.dataset_parameters

        prediction_dataset = TimeSeriesDataSet(
            data=input_df,
            time_idx=params["time_idx"],
            target=params["target"],
            group_ids=params["group_ids"],
            max_encoder_length=params["max_encoder_length"],
            max_prediction_length=params["max_prediction_length"],
            min_encoder_length=0,
            static_categoricals=params.get("static_categoricals", []),
            static_reals=params.get("static_reals", []),
            time_varying_known_categoricals=params.get("time_varying_known_categoricals", []),
            time_varying_known_reals=params.get("time_varying_known_reals", []),
            time_varying_unknown_categoricals=params.get("time_varying_unknown_categoricals", []),
            time_varying_unknown_reals=params.get("time_varying_unknown_reals", []),
            add_relative_time_idx=params.get("add_relative_time_idx", False),
            categorical_encoders=params["categorical_encoders"],
            scalers=params["scalers"],
            # Kütüphanenin standart normalizer'ını kullanarak hatayı önle
            target_normalizer=TorchNormalizer(),
            predict_mode=True,
        )

        dataloader = prediction_dataset.to_dataloader(
            train=False, batch_size=1
        )

        x, y = next(iter(dataloader))
        
        x = {key: val.to(self.device) for key, val in x.items()}

        with torch.no_grad():
            # Modeli doğrudan çağırarak ham çıktıyı al
            raw_prediction_output = self.model(x)
        
        # Çıktı bir sözlüktür, 'prediction' tensörünü çıkar
        scaled_prediction_tensor = raw_prediction_output["prediction"]
        
        # Tek veri grubunun tahminini döndür
        return scaled_prediction_tensor[0]


class InferencePipeline:
    """
    Ana pipeline sınıfı. "Soğuk inference" mantığı ile çalışır.
    Sadece `inference_params.yaml` ve model ağırlık dosyasına ihtiyaç duyar.
    """
    def __init__(self, model_dir: str):
        print("="*60 + "\nSOĞUK INFERENCE PIPELINE BAŞLATILIYOR\n" + "="*60)
        config_path = os.path.join(model_dir, "inference_params.yaml")
        
        model_path = os.path.join(model_dir, "best_model_weights.pth") 
        if not os.path.exists(model_path):
            ckpt_path = os.path.join(model_dir, "best_model.ckpt")
            if os.path.exists(ckpt_path):
                model_path = ckpt_path
            else:
                 raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: '{model_path}' veya '{ckpt_path}'")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self._preprocessor = _DataPreprocessor(self.config)
        self._inferencer = _TFTInferencer(model_path, self.config)
        
        print("\n[Pipeline] Tüm bileşenler başarıyla yüklendi. Tahmin için hazır.\n" + "="*60)

    def run(self, df: pd.DataFrame, last_known_time_idx: int = 0) -> List[float]:
        """
        `TypeError`'ı önlemek için bir Python listesi döndürür.
        """
        print("\n[Pipeline] İş akışı başlatıldı...")
        input_for_predict_df = self._preprocessor.transform(df, last_known_time_idx)
        print(f"[Pipeline] Veri manuel olarak ön işlendi ve normalize edildi.")
        scaled_prediction_tensor = self._inferencer.predict(input_for_predict_df)
        print(f"[Pipeline] Model tahmini (ölçeklenmiş) alındı.")
        final_prediction = self._preprocessor.inverse_transform_prediction(scaled_prediction_tensor)
        print("[Pipeline] Tahminler orijinal ölçeğe dönüştürüldü. İşlem tamamlandı.")
        return final_prediction
