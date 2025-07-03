# test_harness.py
# Bu betik, tüm sistemi MetaTrader 5'ten gelen canlı verilerle
# test etmek için tasarlanmış bir "koşum takımıdır".
# HİÇBİR şekilde alım-satım işlemi gerçekleştirmez.

import MetaTrader5 as mt5
import pandas as pd
import time
import sys
import os
import logging
from dotenv import load_dotenv

# Kendi modüllerimizi import ediyoruz
try:
    from orchestrator import TradingSystemOrchestrator
except ImportError:
    print("HATA: 'orchestrator.py' dosyası bulunamadı. Lütfen aynı dizinde olduğundan emin olun.")
    sys.exit(1)

# --- Loglama Kurulumu ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("test_harness.log"),
        logging.StreamHandler(sys.stdout) # Logları hem dosyaya hem konsola yaz
    ]
)
logger = logging.getLogger(__name__)

# --- .env dosyasından MT5 bilgilerini yükle ---
load_dotenv()
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_TERMINAL_PATH", "")

# --- MT5 Yardımcı Fonksiyonları ---
def mt5_connect():
    """MetaTrader 5'e bağlanır ve terminal bilgilerini doğrular."""
    if not MT5_LOGIN or not MT5_PASSWORD or not MT5_SERVER or not MT5_PATH:
        logger.critical("MT5 bağlantı bilgileri .env dosyasında eksik veya geçersiz.")
        return False
        
    if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, timeout=10000):
        logger.error(f"MT5 initialize() başarısız, hata kodu = {mt5.last_error()}")
        return False
    
    logger.info("MetaTrader 5 bağlantısı başarılı.")
    return True

def mt5_disconnect():
    """MetaTrader 5 bağlantısını sonlandırır."""
    mt5.shutdown()
    logger.info("MetaTrader 5 bağlantısı kapatıldı.")

def get_data_from_mt5(symbol: str, timeframe, count: int) -> pd.DataFrame | None:
    """MT5'ten OHLC verisi çeker ve uygun DataFrame formatına dönüştürür."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"Veri çekilemedi ({symbol}), hata: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        # Sütun isimlerini Orchestrator'ın beklediği hale getir
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        # Gerekli sütunları seçelim
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'time']]
        return df
    except Exception as e:
        logger.error(f"MT5'ten veri çekerken beklenmedik bir hata oluştu: {e}", exc_info=True)
        return None

# --- Ana Test Fonksiyonu ---
def main():
    logger.info("="*50)
    logger.info("      SALT OKUNUR TEST HARNESS BAŞLATILIYOR")
    logger.info("="*50)

    if not mt5_connect():
        sys.exit(1)

    # --- Parametreleri Ayarla ---
    SYMBOL = "EURUSD"  # Test için kullanılacak sembol
    TIMEFRAME_STR = "M1" # Test için 1 dakikalık zaman dilimi
    TIMEFRAME_MT5 = mt5.TIMEFRAME_M1
    MODEL_DIR = "./model_files" # Model dosyalarınızın olduğu klasör

    # --- Orkestratörü Başlat ---
    try:
        # Bu config'ler daha sonra bir YAML dosyasından okunabilir
        feature_config = {
            'ema_short': 9, 'ema_medium': 21, 'ema_long': 50, 'rsi_period': 14,
            'pct_change_multiplier': 1000.0, 'swt_window_size': 32, 'swt_level': 4,
            'ultimate_feature_window': 256
        }
        model_config = {'sequence_length': 128}

        orchestrator = TradingSystemOrchestrator(
            model_dir=MODEL_DIR,
            feature_config=feature_config,
            model_config=model_config
        )
    except Exception as e:
        logger.critical(f"Orkestratör başlatılırken kritik hata: {e}", exc_info=True)
        mt5_disconnect()
        sys.exit(1)

    # --- Tarihsel Veri ile Başlatma ---
    logger.info("Tarihsel veri çekiliyor...")
    historical_df = get_data_from_mt5(SYMBOL, TIMEFRAME_MT5, orchestrator.required_input_rows)
    
    if historical_df is None or len(historical_df) < orchestrator.required_input_rows:
        logger.critical("Başlatma için yeterli tarihsel veri çekilemedi. Bot sonlandırılıyor.")
        mt5_disconnect()
        sys.exit(1)

    orchestrator.initialize_with_historical_data(historical_df)
    
    # Başlangıç tahminini alıp almadığını kontrol et
    try:
        initial_preds = orchestrator.get_prediction()
        logger.info(f"Başlangıç tahmini başarıyla alındı. İlk tahmin değeri: {initial_preds[0]}")
    except Exception as e:
        logger.error(f"Başlangıç tahmini alınırken hata oluştu: {e}", exc_info=True)

    # --- Canlı Veri Akışı Simülasyonu ---
    logger.info("\n" + "-"*50)
    logger.info("      CANLI VERİ AKIŞI TESTİ BAŞLIYOR")
    logger.info("      (Yeni bir mum algılandığında sistem güncellenecek)")
    logger.info("-"*50)

    last_candle_time = historical_df['time'].iloc[-1]

    try:
        while True:
            # En son mumu çek
            latest_candle_data = get_data_from_mt5(SYMBOL, TIMEFRAME_MT5, 1)
            
            if latest_candle_data is not None:
                current_candle_time = latest_candle_data['time'].iloc[0]
                
                # Yeni bir mum oluşmuş mu diye kontrol et
                if current_candle_time > last_candle_time:
                    logger.info("="*20 + f" YENİ MUM ALGILANDI ({pd.to_datetime(current_candle_time, unit='s')}) " + "="*20)
                    last_candle_time = current_candle_time
                    
                    new_candle_series = latest_candle_data.iloc[0]
                    
                    # 1. Orkestratörü güncelle
                    orchestrator.update_with_new_candle(new_candle_series)
                    
                    # 2. Yeni tahmini al
                    try:
                        predictions = orchestrator.get_prediction()
                        logger.info(f"YENİ TAHMİN ALINDI -> İlk Değer: {predictions[0]:.8f}, Son Değer: {predictions[-1]:.8f}")
                    except Exception as e:
                        logger.error(f"Yeni tahmin alınırken bir hata oluştu: {e}", exc_info=True)
                    
                    logger.info("Test adımı tamamlandı. Bir sonraki mum bekleniyor...")

            # CPU'yu yormamak için kısa bir bekleme
            time.sleep(5) 

    except KeyboardInterrupt:
        logger.info("Test kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.critical(f"Ana test döngüsünde beklenmedik bir hata oluştu: {e}", exc_info=True)
    finally:
        mt5_disconnect()

if __name__ == "__main__":
    main()
