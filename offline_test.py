# offline_test.py
# Bu betik, tüm sistemi çevrimdışı bir CSV dosyası üzerinden,
# harici bir bağlantı olmadan test etmek için tasarlanmıştır.
# Çok adımlı tahmin listelerini işler ve bileşik performans ölçümü yapar.

import pandas as pd
import time
import sys
import os
import logging
import numpy as np

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
    logger.info("     ÇOK ADIMLI ÇEVRİMDIŞI TEST VE BİLEŞİK ANALİZ BAŞLATILIYOR")
    logger.info("="*50)

    # --- Parametreleri ve Dosya Yollarını Ayarla ---
    CSV_FILE_PATH = "./data.csv"  # <-- BU YOLU KENDİNİZE GÖRE DEĞİŞTİRİN
    RESULTS_CSV_PATH = "./test_results_compound.csv" # <-- Detaylı sonuçların kaydedileceği dosya
    STEP_SUMMARY_CSV_PATH = "./summary_per_step.csv" # <-- Adım başına özetlerin kaydedileceği dosya
    SPECIAL_STEP_SUMMARY_CSV_PATH = "./summary_per_special_step.csv" # <-- Özel adım başına özetlerin kaydedileceği dosya
    MODEL_DIR = "./core/model"

    # Dosyanın varlığını kontrol et
    if not os.path.exists(CSV_FILE_PATH):
        logger.critical(f"Test verisi dosyası bulunamadı: {CSV_FILE_PATH}")
        sys.exit(1)

    # --- Orkestratörü Başlat ---
    try:
        feature_config = {
            'ema_short': 9, 'ema_medium': 21, 'ema_long': 50, 'rsi_period': 14,
            'pct_change_multiplier': 1000.0, 'swt_window_size': 32, 'swt_level': 4,
            'ultimate_feature_window': 256
        }
        model_config = {'sequence_length': 256}

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
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(full_df.columns):
            logger.critical(f"CSV dosyasında gerekli sütunlar eksik. Beklenen: {required_cols}")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"CSV dosyası okunurken hata: {e}", exc_info=True)
        sys.exit(1)

    initialization_data = full_df.iloc[:orchestrator.required_input_rows]
    live_stream_data = full_df.iloc[orchestrator.required_input_rows:].reset_index(drop=True)
    
    if len(initialization_data) < orchestrator.required_input_rows:
        logger.critical(f"Veri seti, sistemi başlatmak için çok kısa. Gerekli: {orchestrator.required_input_rows}, Mevcut: {len(full_df)}")
        sys.exit(1)

    logger.info(f"Toplam {len(full_df)} satır yüklendi.")
    logger.info(f"{len(initialization_data)} satır ile sistem başlatılacak.")
    logger.info(f"{len(live_stream_data)} satır canlı veri akışı olarak simüle edilecek.")

    # --- Tarihsel Veri ile Başlatma ---
    orchestrator.initialize_with_historical_data(initialization_data)
    
    # --- Canlı Veri Akışı Simülasyonu ---
    logger.info("\n" + "-"*50)
    logger.info("      CANLI VERİ AKIŞI SİMÜLASYONU BAŞLIYOR")
    logger.info("      (Hedef: Gelecekteki mumların yüzdesel değişim serisi)")
    logger.info("-"*50)

    results_list = []
    simulation_steps = min(100, len(live_stream_data) - 1) 
    SPECIAL_STEP_PERIOD = 16 # Her bir tahmin serisini bu periyotlara böl
    
    try:
        for i in range(simulation_steps):
            new_candle_series = live_stream_data.iloc[i]
            
            logger.info("="*10 + f" YENİ MUM İŞLENİYOR (Simülasyon Adım {i+1}/{simulation_steps}) " + "="*10)
            
            orchestrator.update_with_new_candle(new_candle_series)
            
            try:
                predictions = orchestrator.get_prediction()
                prediction_horizon = len(predictions)

                if i + prediction_horizon >= len(live_stream_data):
                    logger.warning(f"Simülasyonun sonuna ulaşıldı, {prediction_horizon} adımlık tahmin serisini karşılaştıracak yeterli veri yok. Bu adım atlanıyor.")
                    break

                next_candles_df = live_stream_data.iloc[i + 1 : i + 1 + prediction_horizon]
                
                actual_opens = next_candles_df['Open']
                actual_closes = next_candles_df['Close']
                actual_pct_changes = ((actual_closes - actual_opens) / actual_opens.replace(to_replace=0, value=1)).fillna(0).tolist()
                
                logger.info(f"TAHMİN LİSTESİ ({prediction_horizon} adım) -> İlk: {predictions[0]:+.6f}, Son: {predictions[-1]:+.6f}")
                logger.info(f"GERÇEK DEĞİŞİM ({prediction_horizon} adım) -> İlk: {actual_pct_changes[0]:+.6f}, Son: {actual_pct_changes[-1]:+.6f}")

                for step in range(prediction_horizon):
                    results_list.append({
                        'simulation_step': i + 1,
                        'prediction_step': step + 1,
                        'special_step': (step // SPECIAL_STEP_PERIOD), # Tahmin serisini 16'lık gruplara ayır
                        'timestamp': next_candles_df.iloc[step].get('Timestamp', pd.to_datetime('now')),
                        'Open': next_candles_df.iloc[step]['Open'],
                        'Close': next_candles_df.iloc[step]['Close'],
                        'predicted_pct_change': (predictions[step])/1000.0,  
                        'actual_pct_change': actual_pct_changes[step]
                    })

            except Exception as e:
                logger.error(f"Yeni tahmin alınırken veya işlenirken bir hata oluştu: {e}", exc_info=True)
            
            time.sleep(0.05)

        logger.info(f"\nSimülasyon başarıyla tamamlandı.")

    except KeyboardInterrupt:
        logger.info("Test kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.critical(f"Ana test döngüsünde beklenmedik bir hata oluştu: {e}", exc_info=True)
    finally:
        if results_list:
            logger.info("\n" + "="*50)
            logger.info("      SONUÇ ANALİZİ VE KAYIT")
            logger.info("="*50)

            results_df = pd.DataFrame(results_list)
            
            # --- METRİK 1: Genel Ortalama Mutlak Hata (Tüm bireysel tahminler üzerinden) ---
            mae = (results_df['actual_pct_change'] - results_df['predicted_pct_change']).abs().mean()
            logger.info(f"Genel Ortalama Mutlak Hata (Tüm Adımlar İçin MAE): {mae:.8f}\n")

            # --- METRİK 2: Tahmin Adımı Başına MAE (Hangi tahmin ufkunun daha iyi olduğunu gösterir) ---
            mae_per_prediction_step = results_df.groupby('prediction_step').apply(
                lambda x: (x['actual_pct_change'] - x['predicted_pct_change']).abs().mean()
            )
            logger.info("Tahmin Adımı Başına MAE Değerleri:")
            for step, mae_val in mae_per_prediction_step.items():
                logger.info(f"  - Tahmin Adımı {step}: {mae_val:.8f}")
            
            logger.info("\n" + "-" * 30)

            # --- METRİK 3: Bileşik Getiri ve MAE Hesaplamaları için Özetleme ---
            
            def calculate_summary_metrics(group):
                """Bir grup tahmin için özet metrikleri hesaplar."""
                mae = (group['actual_pct_change'] - group['predicted_pct_change']).abs().mean()
                # Getiriyi (örn. 0.01) getiri faktörüne (1.01) çevirip çarp, sonra tekrar getiriye çevir
                compound_predicted = (1 + group['predicted_pct_change']).prod() - 1
                compound_actual = (1 + group['actual_pct_change']).prod() - 1
                return pd.Series({
                    'mean_absolute_error': mae,
                    'compound_predicted_return': compound_predicted,
                    'compound_actual_return': compound_actual
                })

            logger.info("      ÖZET METRİK HESAPLAMALARI")
            logger.info("-" * 30)

            # Simülasyon Adımı Başına Özet (Her bir tahmin serisinin genel performansı)
            step_summary_df = results_df.groupby('simulation_step').apply(calculate_summary_metrics)
            logger.info("Simülasyon Adımı Başına Özet Metrikler:")
            with pd.option_context('display.max_rows', 10, 'display.width', 100):
                logger.info(f"\n{step_summary_df.to_string(float_format='%.6f')}")

            # Özel Adım (Special-Step) Başına Özet (Her bir tahmin serisi içindeki 16'lık grupların performansı)
            special_step_summary_df = results_df.groupby(['simulation_step', 'special_step']).apply(calculate_summary_metrics)
            logger.info(f"\nTahmin Serisi İçindeki {SPECIAL_STEP_PERIOD}'lik Periyotlar (Special-Step) Başına Özet Metrikler:")
            with pd.option_context('display.max_rows', 10, 'display.width', 100):
                logger.info(f"\n{special_step_summary_df.to_string(float_format='%.6f')}")


            # --- Sonuçları CSV dosyalarına kaydet ---
            try:
                # Ana, detaylı sonuçlar
                output_columns = ['simulation_step', 'prediction_step', 'special_step', 'timestamp', 'Open', 'Close', 'predicted_pct_change', 'actual_pct_change']
                results_df[output_columns].to_csv(RESULTS_CSV_PATH, index=False, float_format='%.8f')
                logger.info(f"\n\nDetaylı sonuçlar başarıyla '{RESULTS_CSV_PATH}' dosyasına kaydedildi.")
                
                # Özet sonuçlar
                step_summary_df.to_csv(STEP_SUMMARY_CSV_PATH, index=True, float_format='%.8f')
                logger.info(f"Adım başına özet sonuçlar başarıyla '{STEP_SUMMARY_CSV_PATH}' dosyasına kaydedildi.")
                
                special_step_summary_df.to_csv(SPECIAL_STEP_SUMMARY_CSV_PATH, index=True, float_format='%.8f')
                logger.info(f"Özel adım başına özet sonuçlar başarıyla '{SPECIAL_STEP_SUMMARY_CSV_PATH}' dosyasına kaydedildi.")
            except Exception as e:
                logger.error(f"\nSonuçlar CSV dosyasına kaydedilirken hata oluştu: {e}", exc_info=True)
        else:
            logger.warning("Kaydedilecek veya analiz edilecek hiçbir sonuç üretilmedi.")

        logger.info("\nÇevrimdışı test tamamlandı.")

if __name__ == "__main__":
    main()
