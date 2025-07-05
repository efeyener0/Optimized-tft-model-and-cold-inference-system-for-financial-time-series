# orchestrator_final.py

import pandas as pd
import numpy as np
import time
import logging
from typing import Optional, Dict, List

# Kendi kütüphanelerimizi import ediyoruz
from core.feature_engineering_lib import RollingFeaturePipeline
from core.tft_pipeline_core import InferencePipeline

# Loglama için temel yapılandırma
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class TradingSystemOrchestrator:
    """
    Özellik mühendisliği, veri yönetimi ve model tahmini adımlarını
    orkestre eden stateful bir sınıf.
    """
    def __init__(self,
                 model_dir: str,
                 feature_config: Dict,
                 model_config: Dict):
        """
        Orkestratörü başlatır.
        
        Args:
            model_dir (str): TFT modelinin dosyalarının bulunduğu dizin.
            feature_config (Dict): Özellik mühendisliği pipeline'ı için yapılandırma.
            model_config (Dict): Modelin gereksinimleri (örn: sequence_length).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Orkestratör başlatılıyor...")
        
        self.feature_config = feature_config
        self.model_config = model_config
        
        self.feature_pipeline = RollingFeaturePipeline(**self.feature_config)
        self.inference_pipeline = InferencePipeline(model_dir=model_dir)
        
        self.raw_ohlc_data: Optional[pd.DataFrame] = None
        self.processed_feature_data: Optional[pd.DataFrame] = None
        
        # Isınma periyodu, özellik mühendisliğinde NaN oluşumunu önlemek için
        # gereken minimum satır sayısını belirler.
        warmup_values = [
            self.feature_config.get('swt_window_size', 32) - 1,
            self.feature_config.get('ultimate_feature_window', 256) - 1,
            self.feature_config.get('rsi_period', 14),
            self.feature_config.get('ema_long', 50)
        ]
        self.warmup_rows = max(warmup_values)
        self.required_input_rows = self.model_config['sequence_length'] + self.warmup_rows
        
        self.logger.info(f"Özellikler için ısınma periyodu: {self.warmup_rows} satır")
        self.logger.info(f"Tahmin için gerekli toplam satır sayısı: {self.required_input_rows} satır")
        self.logger.info("Orkestratör hazır.")

    def _process_full_chunk(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Verilen bir veri parçasına tüm özellik mühendisliği adımlarını doğru sırada uygular.
        Bu metot, modelin beklediği tüm özellikleri (örn: EMA_9_cA3) oluşturur.
        """
        self.logger.debug(f"Veri parçası işleniyor ({len(df_chunk)} satır)...")
        
        # Adım 1: Temel özellikleri ve göstergeleri hesapla
        processed_df = (df_chunk
            .pipe(self.feature_pipeline.add_percent_change, open_col='Open', close_col='Close')
            .pipe(self.feature_pipeline.add_indicators, close_col='Close')
        )
        
        # Adım 2: Dalgacık dönüşümü uygulanacak tüm sütunları belirle.
        # Bu, hem temel serileri hem de yeni oluşturulan göstergeleri içerir.
        indicator_names = [
            f'EMA_{self.feature_pipeline.ema_short}',
            f'EMA_{self.feature_pipeline.ema_medium}',
            f'EMA_{self.feature_pipeline.ema_long}',
            f'RSI_{self.feature_pipeline.rsi_period}'
        ]
        swt_cols_to_process = ['Close', 'pct_change'] + indicator_names
        
        # Sadece DataFrame'de mevcut olan sütunları işleme al
        swt_cols_to_process = [col for col in swt_cols_to_process if col in processed_df.columns]
        self.logger.debug(f"Dalgacık dönüşümü şu sütunlara uygulanacak: {swt_cols_to_process}")

        # Adım 3: Dalgacık ayrıştırmasını bu genişletilmiş listeye uygula.
        # Bu işlem EMA_9_cA3, RSI_14_cD1 gibi nihai özellikleri oluşturacaktır.
        processed_df = processed_df.pipe(self.feature_pipeline.add_rolling_swt_decomposition, columns_to_process=swt_cols_to_process)
        
        # Adım 4: Nihai özellikleri (enerji, aksiyon proxy) ekle
        processed_df = processed_df.pipe(self.feature_pipeline.add_ultimate_features, open_col='Open', high_col='High', low_col='Low', close_col='Close')
        
        # Adım 5: Tüm işlemlerden sonra oluşan NaN değerleri temizle
        # Bu, özellikle rolling işlemlerin başındaki boşlukları temizler.
        processed_df = processed_df.dropna().reset_index(drop=True)
        
        return processed_df

    def initialize_with_historical_data(self, historical_df: pd.DataFrame):
        """Sistemi, tahminler için yeterli geçmiş veri ile başlatır."""
        if len(historical_df) < self.required_input_rows:
            msg = f"Tarihsel veri en az {self.required_input_rows} satır içermelidir. Sağlanan: {len(historical_df)}"
            self.logger.error(msg)
            raise ValueError(msg)
        
        self.logger.info("Sistem tarihsel veri ile başlatılıyor...")
        self.raw_ohlc_data = historical_df.tail(self.required_input_rows).copy().reset_index(drop=True)
        self.processed_feature_data = self._process_full_chunk(self.raw_ohlc_data)
        
        self.logger.info(f"Başlatma tamamlandı. Hafızada {len(self.raw_ohlc_data)} ham, "
                         f"{len(self.processed_feature_data)} işlenmiş satır bulunuyor.")

    def update_with_new_candle(self, new_candle_series: pd.Series):
        """Sistemi yeni bir mum verisiyle günceller ve özellikleri yeniden hesaplar."""
        if self.raw_ohlc_data is None:
            raise RuntimeError("Sistem önce `initialize_with_historical_data` ile başlatılmalıdır.")
            
        self.logger.info(f"Yeni mum ile güncelleme: Close={new_candle_series.get('Close', 'N/A')}")
        
        # Yeni mumu ekle ve en eski mumu çıkar
        new_candle_df = pd.DataFrame([new_candle_series])
        self.raw_ohlc_data = pd.concat([
            self.raw_ohlc_data.iloc[1:],
            new_candle_df
        ], ignore_index=True)
        
        # Tüm veri parçası üzerinde özellikleri yeniden hesapla
        self.processed_feature_data = self._process_full_chunk(self.raw_ohlc_data)
        self.logger.debug(f"Veri güncellendi. İşlenmiş veri boyutu: {len(self.processed_feature_data)}")

    def get_prediction(self) -> np.ndarray:
        """Mevcut duruma göre modelden bir tahmin alır."""
        if self.processed_feature_data is None:
            raise RuntimeError("Tahmin almadan önce veri işlenmelidir.")
        
        sequence_length = self.model_config['sequence_length']
        model_input_df = self.processed_feature_data.tail(sequence_length)
        
        if len(model_input_df) < sequence_length:
            msg = (f"Model girdisi için yeterli veri yok. `dropna` sonrası satır sayısı azaldı. "
                   f"Beklenen: {sequence_length}, Mevcut: {len(model_input_df)}. "
                   "Daha fazla geçmiş veriyle başlatmayı deneyin.")
            self.logger.warning(msg)
            raise ValueError(msg)
        
        self.logger.info("Tahmin alınıyor...")
        # Gerekli tüm sütunların var olduğundan emin ol
        try:
            predictions = self.inference_pipeline.run(model_input_df)
            self.logger.info("Tahmin başarıyla alındı.")
            return predictions
        except KeyError as e:
            self.logger.error(f"Tahmin sırasında eksik sütun hatası: {e}")
            self.logger.error(f"Modelin beklediği sütunlar: {self.inference_pipeline._inferencer.model.dataset_parameters['time_varying_unknown_reals']}")
            self.logger.error(f"Mevcut sütunlar: {model_input_df.columns.tolist()}")
            raise e
