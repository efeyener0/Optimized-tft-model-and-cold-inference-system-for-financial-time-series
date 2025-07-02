# tft_pipeline_core.py
# Bu modül, bir kütüphane gibi tasarlanmıştır. 
# Sadece yeniden kullanılabilir sınıfları içerir ve çalıştırıldığında bir şey yapmaz.

import os
import yaml
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

# Gerekli kütüphanenin kurulu olduğundan emin olun
try:
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import MAE
except ImportError:
    raise ImportError(
        "Lütfen pytorch-forecasting kütüphanesini kurun: "
        "pip install git+https://github.com/jdb78/pytorch-forecasting.git"
    )

__all__ = ["InferencePipeline"] # `from tft_pipeline_core import *` kullanıldığında sadece bu sınıf import edilir.


class _DataPreprocessor:
    """(Dahili kullanım) Ham veriyi model girdisine ve model çıktısını nihai sonuca dönüştürür."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_def = config['model_definition']
        self.scalers = config['feature_scalers']
        self.target_scaler_info = config['target_scaler']
        self.sequence_length = self.model_def['sequence_length']
        self.target_mean = self.target_scaler_info['params']['mean']
        self.target_std = self.target_scaler_info['params']['std']
        self._extract_variable_lists()

    def _extract_variable_lists(self):
        mdef = self.model_def
        self.static_cats = mdef.get('static_categoricals', [])
        self.time_varying_known_cats = mdef.get('time_varying_known_categoricals', [])
        self.time_varying_unknown_cats = mdef.get('time_varying_unknown_categoricals', [])
        self.group_ids = mdef.get('group_ids', [])
        self.all_cats = self.static_cats + self.time_varying_known_cats + self.time_varying_unknown_cats + self.group_ids

        self.static_reals = mdef.get('static_reals', [])
        self.time_varying_known_reals = mdef.get('time_varying_known_reals', [])
        self.time_varying_unknown_reals = mdef.get('time_varying_unknown_reals', [])
        self.target = mdef['target']
        self.all_reals = self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals
        if self.target not in self.all_reals: self.all_reals.append(self.target)
        self.all_reals.append('relative_time_idx')

    def transform(self, df: pd.DataFrame, device: torch.device) -> Dict[str, torch.Tensor]:
        if len(df) < self.sequence_length:
            raise ValueError(f"Girdi DataFrame'i en az {self.sequence_length} satır olmalıdır. Mevcut: {len(df)}")
        
        input_df = df.tail(self.sequence_length).copy()
        input_df['time_idx'] = np.arange(len(input_df))
        input_df['relative_time_idx'] = input_df['time_idx'].astype(float)
        
        for group_id in self.group_ids:
            if group_id not in input_df.columns:
                input_df[group_id] = "default_group"
        
        for col in self.all_reals:
            if col in self.scalers:
                params = self.scalers[col]['params']
                mean, std = params['mean'], params.get('std', 1.0)
                input_df[col] = (input_df[col] - mean) / std
        
        for col in self.all_cats:
            encoder_key = col if col in self.scalers else f"__group_id__{col}"
            if encoder_key in self.scalers:
                classes = self.scalers[encoder_key]['params']['classes']
                input_df[col] = input_df[col].map(classes).fillna(-1).astype(int)
        
        x_dict = {}
        for col in self.all_cats:
            if col in input_df: x_dict[col] = torch.tensor(input_df[col].values, device=device).long()
        for col in self.all_reals:
            if col in input_df: x_dict[col] = torch.tensor(input_df[col].values, device=device).float()
        
        for key in x_dict:
            x_dict[key] = x_dict[key].unsqueeze(0)
            
        return x_dict

    def inverse_transform_prediction(self, scaled_prediction: torch.Tensor) -> np.ndarray:
        prediction_np = scaled_prediction.squeeze().cpu().numpy()
        return (prediction_np * self.target_std) + self.target_mean


class _TFTInferencer:
    """(Dahili kullanım) Modeli yükler ve tahmin yapar."""
    def __init__(self, model_dir: str, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = os.path.join(model_dir, "best_model_weights.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model ağırlık dosyası bulunamadı: {weights_path}")
        
        self._load_model(weights_path, config)

    def _load_model(self, weights_path: str, config: Dict[str, Any]):
        model_def = config['model_definition']
        model_hyperparams = config['model_hyperparameters']
        categorical_encoders = {k: v for k, v in config['feature_scalers'].items() if v['type'] == 'NaNLabelEncoder'}
        
        all_cats = (model_def.get('static_categoricals', []) +
                    model_def.get('time_varying_known_categoricals', []) +
                    model_def.get('time_varying_unknown_categoricals', []) +
                    model_def.get('group_ids', []))

        categorical_sizes = {}
        for cat in set(all_cats):
            key = cat if cat in categorical_encoders else f"__group_id__{cat}"
            if key in categorical_encoders:
                num_classes = max(categorical_encoders[key]['params']['classes'].values()) + 1
                categorical_sizes[cat] = num_classes
            else:
                raise ValueError(f"'{cat}' için kategorik encoder bilgisi YAML'da bulunamadı.")
        
        full_model_params = {
            **model_hyperparams,
            'categorical_sizes': categorical_sizes,
            'static_categoricals': model_def.get('static_categoricals', []),
            'static_reals': model_def.get('static_reals', []),
            'time_varying_known_categoricals': model_def.get('time_varying_known_categoricals', []),
            'time_varying_known_reals': model_def.get('time_varying_known_reals', []),
            'time_varying_unknown_categoricals': model_def.get('time_varying_unknown_categoricals', []),
            'time_varying_unknown_reals': model_def.get('time_varying_unknown_reals', []),
            'loss': MAE() # Inference'da kullanılmayacak yer tutucu loss
        }
        
        self.model = TemporalFusionTransformer(**full_model_params)
        
        raw_weights = torch.load(weights_path, map_location=self.device)
        state_dict = raw_weights.get('state_dict', raw_weights)
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(clean_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(x_dict)
        return output['prediction']


class InferencePipeline:
    """
    TFT modeli için tam entegre, dosya sisteminden bağımsız tahmin iş akışı.

    Bu sınıf, modeli ve ön işleme adımlarını yükler ve ham bir DataFrame'den
    tahminler üretmek için tek bir arayüz sağlar.
    """
    def __init__(self, model_dir: str):
        """
        Pipeline'ı başlatır.

        Args:
            model_dir (str): 'inference_params.yaml' ve 'best_model_weights.pth' 
                             dosyalarının bulunduğu, kendi kendine yeterli dizin.
        """
        print("="*60 + "\nINFERENCE PIPELINE BAŞLATILIYOR\n" + "="*60)
        config_path = os.path.join(model_dir, "inference_params.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Inference yapılandırma dosyası bulunamadı: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        if 'model_hyperparameters' not in self.config:
            raise KeyError("'inference_params.yaml' dosyasında 'model_hyperparameters' bölümü bulunamadı. "
                           "Lütfen bu bölümü içeren bir config ile modeli yeniden eğitin.")
        
        self._preprocessor = _DataPreprocessor(self.config)
        self._inferencer = _TFTInferencer(model_dir, self.config)
        
        print("\n[Pipeline] Tüm bileşenler başarıyla yüklendi. Tahmin için hazır.\n" + "="*60)

    def run(self, df: pd.DataFrame) -> np.ndarray:
        """
        Verilen ham DataFrame üzerinde tam tahmin iş akışını çalıştırır.

        Args:
            df (pd.DataFrame): Tahmin için kullanılacak, en az 'sequence_length' 
                               uzunluğunda zaman serisi verisi.

        Returns:
            np.ndarray: 'prediction_steps' uzunluğunda, nihai tahmin sonuçları.
        """
        print("\n[Pipeline] İş akışı başlatıldı...")
        x_dict = self._preprocessor.transform(df, self._inferencer.device)
        print("[Pipeline] Veri ön işlendi ve tensörlere dönüştürüldü.")
        
        scaled_prediction = self._inferencer.predict(x_dict)
        print("[Pipeline] Model tahmini (ölçeklenmiş) alındı.")
        
        final_prediction = self._preprocessor.inverse_transform_prediction(scaled_prediction)
        print("[Pipeline] Tahminler orijinal ölçeğe dönüştürüldü. İşlem tamamlandı.")
        
        return final_prediction
