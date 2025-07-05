# --- START OF FINAL packager.py ---

import os
import sys
import yaml
import shutil
import argparse
import inspect
import traceback
import subprocess  # pip freeze çalıştırmak için eklendi
from pathlib import Path

# Kural 1.1: Eğitim ortamını tanımlayan modülleri doğrudan import et.
# Bu, olası sapmaları imkansız hale getirir.
try:
    from trainer import GpuPreloadedDataModule, LightningTFT
except ImportError as e:
    print(f"KRİTİK HATA: 'trainer.py' dosyası bulunamadı veya import edilemiyor: {e}", file=sys.stderr)
    print("Lütfen bu script'i 'trainer.py' ile aynı dizinde çalıştırdığınızdan emin olun.", file=sys.stderr)
    sys.exit(1)

import torch
import pytorch_forecasting
import pandas
import sklearn


class ModelPackager:
    """
    Eğitilmiş bir modeli ve tüm kod bağımlılıklarını tek bir, taşınabilir
    ve yeniden üretilebilir "deployment_package" klasörüne paketler.

    Felsefe: "Sıfır Güven". Her adım doğrulanır, her bağımlılık sabitlenir.
    """
    def __init__(self, config_path: str, run_dir: str):
        self.config_path = Path(config_path)
        self.run_dir = Path(run_dir)
        
        # Paranoyak Başlangıç Kontrolleri
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Yapılandırma dosyası bulunamadı veya bir dizin: {self.config_path}")
        if not self.run_dir.is_dir():
            raise NotADirectoryError(f"Çalışma yolu bir dizin değil: {self.run_dir}")

        self.weights_path = self.run_dir / 'best_model_weights.pth'
        if not self.weights_path.is_file():
             raise FileNotFoundError(f"En iyi model ağırlıkları bulunamadı: {self.weights_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            if not isinstance(self.config, dict) or 'model' not in self.config or 'data' not in self.config:
                raise ValueError("Yapılandırma dosyası 'model' ve/veya 'data' anahtarlarını içermiyor veya formatı bozuk.")
        except (yaml.YAMLError, IOError, ValueError) as e:
            raise IOError(f"Yapılandırma dosyası okunamadı, bozuk veya geçersiz: {e}") from e

        print("Model Paketleyici başlatıldı.")
        print(f"  - Yapılandırma: {self.config_path}")
        print(f"  - Çalışma Dizini: {self.run_dir}")

    def _clone_environment_and_extract_artifacts(self) -> dict:
        print("\n[Adım 1/4] Eğitim ortamı klonlanıyor ve artifaktlar çıkarılıyor...")
        try:
            print("  - Veri modülü oluşturuluyor...")
            datamodule = GpuPreloadedDataModule(self.config)
            datamodule.setup(stage='fit')

            print("  - Boş model mimarisi oluşturuluyor...")
            lightning_model = LightningTFT(**self.config)
            lightning_model.setup(stage='fit')
            tft_model = lightning_model.model

            print(f"  - Ağırlıklar '{self.weights_path}' dosyasından yükleniyor...")
            state_dict = torch.load(self.weights_path, map_location='cpu')
            tft_model.load_state_dict(state_dict)

            payload = {
                'model_state_dict': tft_model.state_dict(),
                'model_hyperparameters': tft_model.hparams,
                'dataset_params': datamodule.training_ts_dataset.get_parameters()
            }
            print("  - Artifaktlar (ağırlıklar, hparamlar, veri şeması) başarıyla toplandı.")
            return payload
        except Exception as e:
            print(f"HATA: Eğitim ortamı klonlanırken bir hata oluştu: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def _copy_dependencies(self, target_lib_dir: Path):
        print("\n[Adım 2/4] Kod bağımlılıkları kopyalanıyor...")
        
        # DİKKAT: Eğer 'trainer.py' yeni yerel modüller (örn: utils.py) import etmeye başlarsa,
        # bu modüllerin de buraya manuel olarak eklenmesi gerekir.
        dependencies_to_copy = {
            "pytorch_forecasting": pytorch_forecasting,
            "trainer": sys.modules.get('trainer')
        }

        for name, module in dependencies_to_copy.items():
            if module is None:
                print(f"  - UYARI: '{name}' modülü bulunamadı, atlanıyor.", file=sys.stderr)
                continue
            try:
                module_path = Path(inspect.getfile(module))
                if module_path.name == '__init__.py':
                    source_dir = module_path.parent
                    target_dir = target_lib_dir / source_dir.name
                    if target_dir.exists(): shutil.rmtree(target_dir)
                    shutil.copytree(source_dir, target_dir)
                else:
                    source_dir = module_path
                    target_dir = target_lib_dir / source_dir.name
                    shutil.copy2(source_dir, target_dir)
                
                if not target_dir.exists():
                    raise IOError(f"Kopyalama sonrası hedef '{target_dir}' bulunamadı.")
                print(f"  - '{name}' bağımlılığı başarıyla kopyalandı.")

            except Exception as e:
                print(f"HATA: '{name}' bağımlılığı kopyalanamadı: {e}", file=sys.stderr)
                traceback.print_exc()
                raise

    def _create_requirements_file(self, target_file: Path):
        print("\n[Adım 3/4] Tam bağımlılık listesi ('requirements.txt') oluşturuluyor...")
        print("  - Mevcut Python ortamının tam bir kopyası için 'pip freeze' kullanılıyor.")
        
        try:
            # Mevcut betiği çalıştıran python yorumlayıcısını kullanarak pip'i çağırmak en güvenli yoldur.
            # Bu, doğru sanal ortamın (virtualenv, venv, conda) kullanılmasını sağlar.
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            
            print(f"  - Başarılı: {len(result.stdout.splitlines())} paket 'requirements.txt' dosyasına yazıldı.")
            
        except FileNotFoundError:
            print("HATA: 'pip' komutu mevcut Python ortamında bulunamadı.", file=sys.stderr)
            print("Lütfen Python kurulumunuzun sağlam olduğundan ve 'pip' modülünün erişilebilir olduğundan emin olun.", file=sys.stderr)
            raise
        except subprocess.CalledProcessError as e:
            print(f"HATA: 'pip freeze' komutu çalıştırılırken bir hata oluştu (çıkış kodu {e.returncode}):", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            raise
        except Exception as e:
            print(f"HATA: 'requirements.txt' oluşturulurken beklenmedik bir hata oluştu: {e}", file=sys.stderr)
            traceback.print_exc()
            raise


    def package(self, output_dir: str):
        output_path = Path(output_dir)
        lib_path = output_path / 'lib'

        try:
            if output_path.exists():
                print(f"Mevcut çıktı dizini '{output_path}' temizleniyor...")
                shutil.rmtree(output_path)
            
            output_path.mkdir(parents=True, exist_ok=True)
            lib_path.mkdir(exist_ok=True)
        except OSError as e:
            print(f"HATA: Çıktı dizini oluşturulamadı/temizlenemedi: {e}", file=sys.stderr)
            print("Lütfen dosya sistemi izinlerinizi kontrol edin.", file=sys.stderr)
            raise

        payload = self._clone_environment_and_extract_artifacts()
        self._copy_dependencies(lib_path)
        self._create_requirements_file(output_path / 'requirements.txt')

        print("\n[Adım 4/4] Son paket dosyaları oluşturuluyor...")
        payload_path = output_path / 'packaged_model.pth'
        torch.save(payload, payload_path)
        print(f"  - Model artifaktları şuraya kaydedildi: {payload_path}")

        predictor_script_content = self._get_predictor_script()
        predictor_path = output_path / 'predictor.py'
        with open(predictor_path, 'w', encoding='utf-8') as f:
            f.write(predictor_script_content)
        print(f"  - Çalıştırıcı script şuraya oluşturuldu: {predictor_path}")
        
        print("\n" + "="*50)
        print("PAKETLEME BAŞARIYLA TAMAMLANDI!")
        print(f"Dağıtıma hazır paket şurada: '{output_path.resolve()}'")
        print("\nKullanım için:")
        print(f"  1. cd {output_path.resolve()}")
        print("  2. (Öneri) python -m venv venv && source venv/bin/activate")
        print("  3. pip install -r requirements.txt")
        print("  4. python predictor.py")
        print("="*50)

    @staticmethod
    def _get_predictor_script() -> str:
        # Gömülecek olan 'predictor.py'nin en son, en sağlam hali
        return """
# --- START OF EMBEDDED predictor.py ---
# Bu script, ana paketleyici tarafından otomatik olarak oluşturulmuştur.

import sys
import os
import traceback
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# --- Güvenli Yerel Kütüphane Yükleyicisi ---
try:
    lib_path = Path(__file__).parent / 'lib'
    if lib_path.is_dir() and str(lib_path) not in sys.path:
        print(f"[Predictor] Yerel kütüphane yolu '{lib_path}' sisteme ekleniyor.")
        sys.path.insert(0, str(lib_path))
except Exception as e:
    print(f"KRİTİK HATA: Yerel kütüphane yolu eklenemedi: {e}", file=sys.stderr)
    sys.exit(1)

# --- Güvenli Importlar ---
try:
    from sklearn.preprocessing import StandardScaler
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer, EncoderNormalizer
    from pytorch_forecasting import TimeSeriesDataSet
except ImportError as e:
    print(f"KRİTİK HATA: Gerekli kütüphaneler yüklenemedi.", file=sys.stderr)
    print("Paketin 'lib' klasörünün sağlam olduğundan ve 'requirements.txt' dosyasının kurulduğundan emin olun.", file=sys.stderr)
    print(f"Orijinal Hata: {e}", file=sys.stderr)
    sys.exit(1)


class Predictor:
    def __init__(self, artifact_path: str = "packaged_model.pth"):
        print("[Predictor] Başlatılıyor...")
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  - Cihaz: {self.device}")

            payload = torch.load(artifact_path, map_location=self.device)
            print(f"  - Artifakt '{artifact_path}' yüklendi.")

            # Gerekli bilgileri doğrulamalı bir şekilde paketten çıkar
            self.dataset_params = payload.get('dataset_params')
            self.model_hyperparameters = payload.get('model_hyperparameters')
            model_state_dict = payload.get('model_state_dict')
            if not all([self.dataset_params, self.model_hyperparameters, model_state_dict]):
                 raise ValueError("Artifakt dosyası eksik anahtarlar içeriyor: 'dataset_params', 'model_hyperparameters', 'model_state_dict' gerekli.")

            self.model_definition = self.dataset_params.get('model_definition', {})

            print("  - Model mimarisi yeniden oluşturuluyor...")
            self.model = TemporalFusionTransformer.from_parameters(**self.model_hyperparameters)
            
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("  - Model ağırlıkları yüklendi ve tahmin moduna alındı.")

            print("  - Veri dönüştürücüler (scaler) yeniden oluşturuluyor...")
            self.scalers = self._recreate_scalers()
            self.is_ready = True
            print("[Predictor] Kurulum tamamlandı. Tahmin için hazır.")
        except Exception as e:
            print(f"HATA: Predictor başlatılırken bir hata oluştu: {e}", file=sys.stderr)
            traceback.print_exc()
            self.is_ready = False
            raise

    def _recreate_scalers(self) -> dict:
        recreated_scalers = {}
        
        # Pytorch-forecasting > 0.10.1 için uyumluluk
        if 'target_normalizer' in self.dataset_params:
            self.dataset_params.setdefault('scalers', {})
            target_col = self.dataset_params['target']
            self.dataset_params['scalers'][target_col] = self.dataset_params['target_normalizer']
        
        # Savunmacı scaler yeniden oluşturma
        for col, scaler_obj in self.dataset_params.get('scalers', {}).items():
            params = {}
            if isinstance(scaler_obj, (TorchNormalizer, EncoderNormalizer)) or hasattr(scaler_obj, 'get_parameters'):
                params = scaler_obj.get_parameters()

            if isinstance(params, dict) and "mean_" in params and "scale_" in params:
                try:
                    scaler = StandardScaler()
                    scaler.mean_ = np.array(params['mean_'], dtype=np.float64)
                    scaler.scale_ = np.array(params['scale_'], dtype=np.float64)
                    recreated_scalers[col] = scaler
                except (ValueError, TypeError) as e:
                    print(f"UYARI: '{col}' için scaler parametreleri geçersiz: {e}", file=sys.stderr)
        
        for col, encoder_obj in self.dataset_params.get('categorical_encoders', {}).items():
            if isinstance(encoder_obj, NaNLabelEncoder):
                 recreated_scalers[col] = encoder_obj

        if not recreated_scalers:
             print("UYARI: Hiçbir scaler yeniden oluşturulamadı. Veri normalizasyonu yapılmayabilir.", file=sys.stderr)
        return recreated_scalers

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError("Predictor kurulumu başarısız olduğu için hazır değil.")
        
        print("\\n[Tahmin Süreci] Başlatılıyor...")
        try:
            min_len = self.model_definition['max_encoder_length']
            if len(df) < min_len:
                raise ValueError(f"Girdi DataFrame'i en az {min_len} satır içermelidir, ancak {len(df)} satır var.")

            prediction_dataset = TimeSeriesDataSet.from_dataset(
                self.dataset_params, df, predict=True, stop_randomization=True
            )
            prediction_loader = prediction_dataset.to_dataloader(batch_size=1)
            print("  - Girdi verisi, kütüphanenin kendi metoduyla güvenli bir şekilde model formatına dönüştürüldü.")

            with torch.no_grad():
                raw_predictions, _ = self.model.predict(prediction_loader, mode="raw", return_x=True)
            print("  - Ham tahminler (normalize) alındı.")

            target_col = self.dataset_params['target']
            target_scaler = self.scalers.get(target_col)

            if not isinstance(target_scaler, StandardScaler):
                print(f"UYARI: Hedef değişken için StandardScaler bulunamadı. Sonuçlar normalize edilmiş olabilir.", file=sys.stderr)
                return raw_predictions['prediction'].cpu().numpy().squeeze()

            predictions_scaled = raw_predictions['prediction'].cpu().numpy()
            inversed_predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
            
            print("  - Tahminler orijinal ölçeğe çevrildi.")
            return inversed_predictions.flatten()
            
        except Exception as e:
            print(f"HATA: Tahmin sırasında bir hata oluştu: {e}", file=sys.stderr)
            expected_cols = list(self.dataset_params.get('reals', [])) + list(self.dataset_params.get('categoricals', []))
            print("Girdi DataFrame'inizin modelin eğitim verisiyle aynı sütunlara ve veri tiplerine sahip olduğundan emin olun.", file=sys.stderr)
            print(f"Beklenen sütunlardan bazıları: {expected_cols[:5]}...", file=sys.stderr)
            traceback.print_exc()
            raise

if __name__ == '__main__':
    try:
        predictor = Predictor()
        
        print(f"\\n--- Örnek Tahmin Senaryosu ---")
        seq_len = predictor.model_definition['max_encoder_length']
        pred_len = predictor.model_definition['max_prediction_length']
        
        # Modelin beklediği tüm sütunları programatik olarak al
        all_cols = set()
        for key in ['reals', 'categoricals', 'group_ids', 'target']:
            all_cols.update(predictor.dataset_params.get(key, []))

        mock_data = {}
        for col in all_cols:
            if col in predictor.dataset_params.get('reals', []) or col == predictor.dataset_params['target']:
                 mock_data[col] = np.random.randn(seq_len)
            else: # Kategorik veya grup ID'si
                 # İYİLEŞTİRME: Modelin bildiği gerçek bir kategoriyi kullanmayı dene, başarısız olursa sahte bir kategoriye geri dön.
                 try:
                     encoder = predictor.dataset_params['categorical_encoders'][col]
                     mock_data[col] = encoder.classes_[0]
                 except (KeyError, IndexError, AttributeError):
                     mock_data[col] = f'cat_{col}_A' # Fallback

        # Statik özellikleri ele al
        for col in predictor.dataset_params.get('static_reals', []):
            mock_data[col] = np.random.randn()
        for col in predictor.dataset_params.get('static_categoricals', []):
             try:
                 encoder = predictor.dataset_params['categorical_encoders'][col]
                 mock_data[col] = encoder.classes_[0]
             except (KeyError, IndexError, AttributeError):
                 mock_data[col] = 'static_cat_A' # Fallback

        # time_idx zorunludur
        if 'time_idx' not in mock_data: mock_data['time_idx'] = range(seq_len)
        
        input_df = pd.DataFrame(mock_data)
        print(f"Örnek girdi verisi oluşturuldu. Boyut: {input_df.shape}, Sütunlar: {list(input_df.columns)}")

        predictions = predictor.predict(input_df)

        print("\\n--- TAHMİN SONUÇLARI ---")
        print(f"Gelecek {pred_len} adım için yapılan tahminler:")
        for i, p in enumerate(predictions):
            print(f"  Adım {i+1}: {p:.4f}")

    except Exception:
        print("\\nProgram bir hata nedeniyle sonlandırıldı.", file=sys.stderr)
        sys.exit(1)

# --- END OF EMBEDDED predictor.py ---
"""

# Ana script'in __main__ bloğu
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TFT modelini ve tüm bağımlılıklarını taşınabilir bir klasöre paketler.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Modeli eğitmek için kullanılan orijinal 'config.yaml' dosyasının yolu."
    )
    parser.add_argument(
        '--run_dir', type=str, required=True,
        help="Eğitim sonrası oluşturulan ve 'best_model_weights.pth' dosyasını içeren dizin."
    )
    parser.add_argument(
        '--output_dir', type=str, default='deployment_package',
        help="Paketlenmiş dosyaların kaydedileceği klasörün adı. (Varsayılan: deployment_package)"
    )
    args = parser.parse_args()

    try:
        packager = ModelPackager(config_path=args.config, run_dir=args.run_dir)
        packager.package(output_dir=args.output_dir)
    except Exception as e:
        print(f"\n!!! PAKETLEME SIRASINDA KRİTİK BİR HATA OLUŞTU !!!", file=sys.stderr)
        # Hata zaten ilgili fonksiyonda loglandığı için burada sadece genel bir mesaj veriyoruz.
        # traceback.print_exc() # Gerekirse daha fazla detay için açılabilir.
        sys.exit(1)

# --- END OF FINAL packager.py ---
