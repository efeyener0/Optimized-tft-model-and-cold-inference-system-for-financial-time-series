# offline_test.py
# Bu betik, tüm sistemi çevrimdışı bir CSV dosyası üzerinden,
# harici bir bağlantı olmadan test etmek için tasarlanmıştır.
# Bir canlı veri akışını simüle eder.

import pandas as pd
import time
import sys
import os
import logging

# Kendi modüllerimizi import ediyoruz
try:
    from core.orchestrator import TradingSystemOrchestrator
except ImportError:
    print("HATA: 'orchestrator.py' dosyası bulunamadı. Lütfen aynı dizinde olduğundan emin olun.")
    sys.exit(1)

# --- Loglama Kurulumu ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("offline_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Ana Test Fonksiyonu ---
def main():
    logger.info("="*50)
    logger.info("      ÇEVRİMDIŞI CSV TESTİ BAŞLATILIYOR")
    logger.info("="*50)

    # --- Parametreleri ve Dosya Yollarını Ayarla ---
    CSV_FILE_PATH = "path/to/your/large_historical_data.csv"  # <-- BU YOLU KENDİNİZE GÖRE DEĞİŞTİRİN
    MODEL_DIR = "./model_files"
    
    # Dosyanın varlığını kontrol et
    if not os.path.exists(CSV_FILE_PATH):
        logger.critical(f"Test verisi dosyası bulunamadı: {CSV_FILE_PATH}")
        logger.critical("Lütfen doğru dosya yolunu belirttiğinizden emin olun.")
        sys.exit(1)

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
        sys.exit(1)

    # --- Veriyi Yükle ve Başlatma İçin Ayır ---
    logger.info(f"Test verisi yükleniyor: {CSV_FILE_PATH}")
    try:
        full_df = pd.read_csv(CSV_FILE_PATH)
        # Sütun isimlerinin doğru olduğundan emin olalım (varsayılanlar)
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(full_df.columns):
            logger.critical(f"CSV dosyasında gerekli sütunlar eksik. Beklenen: {required_cols}")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"CSV dosyası okunurken hata: {e}", exc_info=True)
        sys.exit(1)

    # Veriyi, başlatma ve canlı akış simülasyonu için ikiye ayır
    initialization_data = full_df.iloc[:orchestrator.required_input_rows]
    live_stream_data = full_df.iloc[orchestrator.required_input_rows:]
    
    if len(initialization_data) < orchestrator.required_input_rows:
        logger.critical(f"Veri seti, sistemi başlatmak için çok kısa. Gerekli: {orchestrator.required_input_rows}, Mevcut: {len(full_df)}")
        sys.exit(1)

    logger.info(f"Toplam {len(full_df)} satır yüklendi.")
    logger.info(f"{len(initialization_data)} satır ile sistem başlatılacak.")
    logger.info(f"{len(live_stream_data)} satır canlı veri akışı olarak simüle edilecek.")

    # --- Tarihsel Veri ile Başlatma ---
    orchestrator.initialize_with_historical_data(initialization_data)
    
    # Başlangıç tahminini alıp almadığını kontrol et
    try:
        initial_preds = orchestrator.get_prediction()
        logger.info(f"Başlangıç tahmini başarıyla alındı. İlk tahmin değeri: {initial_preds[0]}")
    except Exception as e:
        logger.error(f"Başlangıç tahmini alınırken hata oluştu: {e}", exc_info=True)

    # --- Canlı Veri Akışı Simülasyonu ---
    logger.info("\n" + "-"*50)
    logger.info("      CANLI VERİ AKIŞI SİMÜLASYONU BAŞLIYOR")
    logger.info("      (CSV'deki her yeni satır, yeni bir mum olarak işlenecek)")
    logger.info("-"*50)

    # Sadece ilk 100 mumu simüle edelim (tüm dosyayı değil)
    simulation_steps = min(100, len(live_stream_data))
    
    try:
        for i in range(simulation_steps):
            new_candle_series = live_stream_data.iloc[i]
            
            logger.info("="*20 + f" YENİ MUM İŞLENİYOR (Simülasyon Adım {i+1}/{simulation_steps}) " + "="*20)
            
            # 1. Orkestratörü güncelle
            orchestrator.update_with_new_candle(new_candle_series)
            
            # 2. Yeni tahmini al
            try:
                predictions = orchestrator.get_prediction()
                logger.info(f"YENİ TAHMİN ALINDI -> İlk Değer: {predictions[0]:.8f}, Son Değer: {predictions[-1]:.8f}")
            except Exception as e:
                logger.error(f"Yeni tahmin alınırken bir hata oluştu: {e}", exc_info=True)
            
            time.sleep(0.1) # Simülasyonu biraz yavaşlatarak okunabilirliği artır

        logger.info(f"\n{simulation_steps} adımlık simülasyon başarıyla tamamlandı.")

    except KeyboardInterrupt:
        logger.info("Test kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.critical(f"Ana test döngüsünde beklenmedik bir hata oluştu: {e}", exc_info=True)
    finally:
        logger.info("Çevrimdışı test tamamlandı.")

if __name__ == "__main__":
    main()
