# feature_engineering_lib.py
# Bu modül, bir "araç kutusu" olarak tasarlanmıştır. Her özellik hesaplama
# fonksiyonu bağımsızdır ve dışarıdan bir orkestratör (wrapper) tarafından
# istenen sırada çağrılmak üzere tasarlanmıştır.

import pandas as pd
import numpy as np
from typing import List, Dict

try:
    import pywt
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    raise ImportError("Lütfen PyWavelets ve NumPy kütüphanelerini kurun: 'pip install PyWavelets numpy'")

__all__ = ["RollingFeaturePipeline"]


class RollingFeaturePipeline:
    """
    Büyük zaman serileri için yapılandırılabilir ve verimli bir ROLLING özellik mühendisliği araç kutusu.
    Her metot, bir DataFrame alıp, ilgili özellikleri eklenmiş yeni bir DataFrame döndürür.
    """
    def __init__(self,
                 ema_short: int = 9,
                 ema_medium: int = 21,
                 ema_long: int = 50,
                 rsi_period: int = 14,
                 pct_change_multiplier: float = 1000.0,
                 swt_window_size: int = 32,
                 swt_wavelet: str = 'db4',
                 swt_level: int = 4,
                 ultimate_feature_window: int = 256,
                 ultimate_swt_window: int = 32,
                 ultimate_swt_level: int = 3,
                 ultimate_wavelet: str = 'db4'
                 ):
        """Pipeline araçlarını belirtilen parametrelerle yapılandırır."""
        self.ema_short, self.ema_medium, self.ema_long, self.rsi_period = ema_short, ema_medium, ema_long, rsi_period
        self.pct_change_multiplier = pct_change_multiplier
        self.swt_window_size, self.swt_wavelet = swt_window_size, swt_wavelet
        self.ultimate_feature_window, self.ultimate_swt_window, self.ultimate_swt_level, self.ultimate_wavelet = \
            ultimate_feature_window, ultimate_swt_window, ultimate_swt_level, ultimate_wavelet
        
        max_level_swt = pywt.swt_max_level(self.swt_window_size)
        self.swt_level = min(swt_level, max_level_swt)
        if swt_level > max_level_swt:
            print(f"Uyarı: SWT Seviyesi ({swt_level}) pencere boyutu ({self.swt_window_size}) için çok yüksek. "
                  f"Maksimum olası seviye ({self.swt_level}) olarak ayarlandı.")
        
        self._wavelet_filter_cache: Dict[str, Dict[str, np.ndarray]] = {}

    # --- 1. Yardımcı Metotlar (Dahili Kullanım) ---
    def _compute_rsi_vectorized(self, data: pd.Series) -> pd.Series:
        delta = data.diff()
        gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _get_wavelet_filters(self, wavelet_name: str) -> Dict[str, np.ndarray]:
        if wavelet_name not in self._wavelet_filter_cache:
            wave = pywt.Wavelet(wavelet_name)
            self._wavelet_filter_cache[wavelet_name] = {
                'lo': np.array(wave.dec_lo)[::-1].reshape(1, 1, -1),
                'hi': np.array(wave.dec_hi)[::-1].reshape(1, 1, -1)
            }
        return self._wavelet_filter_cache[wavelet_name]

    # --- 2. Bağımsız Özellik Mühendisliği "Araçları" ---
    
    @staticmethod
    def delete_first_n_rows(df: pd.DataFrame, n_rows_to_delete: int) -> pd.DataFrame:
        """Bir DataFrame'in ilk N satırını siler."""
        if n_rows_to_delete <= 0: return df
        if n_rows_to_delete >= len(df): return df.iloc[0:0]
        return df.iloc[n_rows_to_delete:].reset_index(drop=True)

    def add_indicators(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """DataFrame'e EMA ve RSI göstergelerini ekler."""
        df_out = df.copy()
        if close_col not in df_out.columns: raise KeyError(f"'{close_col}' sütunu DataFrame'de bulunamadı.")
        df_out[f'EMA_{self.ema_short}'] = df_out[close_col].ewm(span=self.ema_short, adjust=False).mean()
        df_out[f'EMA_{self.ema_medium}'] = df_out[close_col].ewm(span=self.ema_medium, adjust=False).mean()
        df_out[f'EMA_{self.ema_long}'] = df_out[close_col].ewm(span=self.ema_long, adjust=False).mean()
        df_out[f'RSI_{self.rsi_period}'] = self._compute_rsi_vectorized(df_out[close_col])
        return df_out
        
    def add_percent_change(self, df: pd.DataFrame, open_col: str, close_col: str) -> pd.DataFrame:
        """Açılış ve Kapanış fiyatları arasındaki yüzde değişimi hesaplar."""
        df_out = df.copy()
        if open_col not in df_out.columns or close_col not in df_out.columns:
            raise KeyError(f"'{open_col}' veya '{close_col}' sütunları bulunamadı.")
        df_out['pct_change'] = ((df_out[close_col] - df_out[open_col]) / (df_out[open_col] + 1e-9)) * self.pct_change_multiplier
        return df_out

    def add_rolling_swt_decomposition(self, df: pd.DataFrame, columns_to_process: List[str]) -> pd.DataFrame:
        """Belirtilen sütunlara kayan pencere ile SWT uygular."""
        df_out = df.copy()
        for col_name in columns_to_process:
            if col_name not in df_out.columns:
                print(f"Uyarı: Rolling SWT için '{col_name}' sütunu bulunamadı, atlanıyor.")
                continue
            signal = df_out[col_name].to_numpy(dtype=np.float32)
            if len(signal) < self.swt_window_size:
                print(f"Uyarı: '{col_name}' serisi ({len(signal)}) pencere boyutundan ({self.swt_window_size}) kısa. Atlanıyor.")
                continue
            
            windows = sliding_window_view(signal, window_shape=self.swt_window_size)
            
            def process_window(w):
                # Dalgacık ayrıştırmasını try-except bloğu içinde güvenli bir şekilde yap
                try:
                    # NİHAİ DÜZELTME: trim_approx=True parametresini kaldırarak
                    # coeffs'in her zaman bir tuple listesi [(cA, cD), ...]
                    # formatında olmasını garanti et. Bu, indeksleme mantığını
                    # tutarlı ve hatasız hale getirir.
                    coeffs = pywt.swt(w, self.swt_wavelet, level=self.swt_level)
                    
                    # Dönen sonucun beklenen formatta olup olmadığını kontrol et
                    if not isinstance(coeffs, list) or len(coeffs) < self.swt_level:
                        return (np.nan,) * 5
                except Exception as e:
                    # SWT sırasında herhangi bir hata olursa, NaN döndür
                    # print(f"SWT hatası: {e}") # Hata ayıklama için
                    return (np.nan,) * 5

                # Artık coeffs'in geçerli bir tuple listesi olduğundan eminiz.
                c_a3 = coeffs[self.swt_level-3][0][-1] if self.swt_level >= 3 else np.nan
                c_d4 = coeffs[self.swt_level-4][1][-1] if self.swt_level >= 4 else np.nan
                c_d3 = coeffs[self.swt_level-3][1][-1] if self.swt_level >= 3 else np.nan
                c_d2 = coeffs[self.swt_level-2][1][-1] if self.swt_level >= 2 else np.nan
                c_d1 = coeffs[self.swt_level-1][1][-1] if self.swt_level >= 1 else np.nan
                return c_a3, c_d1, c_d2, c_d3, c_d4

            results = np.apply_along_axis(process_window, 1, windows)
            
            start_index = self.swt_window_size - 1
            df_out.loc[start_index:, f'{col_name}_cA3'] = results[:, 0]
            df_out.loc[start_index:, f'{col_name}_cD1'] = results[:, 1]
            df_out.loc[start_index:, f'{col_name}_cD2'] = results[:, 2]
            df_out.loc[start_index:, f'{col_name}_cD3'] = results[:, 3]
            df_out.loc[start_index:, f'{col_name}_cD4'] = results[:, 4]
        return df_out

    def add_ultimate_features(self, df: pd.DataFrame, open_col: str, high_col: str, low_col: str, close_col: str) -> pd.DataFrame:
        """Enerji ve aksiyon proxy özelliklerini üretir."""
        df_out = df.copy()
        epsilon = 1e-12
        z_score_cols = {col: ((df_out[col] - df_out[col].rolling(self.ultimate_feature_window, min_periods=1).mean()) / 
                              (df_out[col].rolling(self.ultimate_feature_window, min_periods=1).std() + epsilon) + 1.0) / 2.0
                        for col in [open_col, high_col, low_col, close_col]}
        for col, z_series in z_score_cols.items(): z_score_cols[col] = z_series.clip(0, 1).fillna(0.5)

        price_z = (z_score_cols[open_col] + z_score_cols[high_col] + z_score_cols[low_col] + z_score_cols[close_col]) / 4.0

        if len(price_z) >= self.ultimate_swt_window:
            windows = sliding_window_view(price_z.to_numpy(dtype=np.float64), self.ultimate_swt_window)
            filters = self._get_wavelet_filters(self.ultimate_wavelet)
            energy_val = np.zeros(len(price_z) - self.ultimate_swt_window + 1, dtype=np.float64)
            current_coeffs = windows.copy()
            for level_idx in range(self.ultimate_swt_level):
                dilation = 2 ** level_idx; hi = filters['hi'].flatten()
                hi_dilated = np.zeros((hi.size - 1) * dilation + 1); hi_dilated[::dilation] = hi
                detail_coeffs = np.apply_along_axis(lambda m: np.convolve(m, hi_dilated, mode='same'), 1, current_coeffs)
                energy_val += np.abs(detail_coeffs).sum(axis=1)
            detail_energy = np.zeros(len(df_out)); detail_energy[self.ultimate_swt_window - 1:] = energy_val
            df_out['feat_energy_final'] = np.cbrt(np.log(detail_energy + epsilon))
        else: df_out['feat_energy_final'] = 0.0

        closer = np.where((z_score_cols[close_col] - z_score_cols[low_col]).abs() < (z_score_cols[close_col] - z_score_cols[high_col]).abs(), 0.0, 1.0)
        width = np.where(closer == 0.0, (price_z - z_score_cols[low_col]).abs(), (z_score_cols[high_col] - price_z).abs())
        height = 0.5 * (price_z + price_z.shift(1).fillna(price_z.iloc[0]))
        integrand = (height + 1.0) * (width + 1.0) - 1.0
        df_out['feat_action_proxy'] = np.sqrt(integrand.astype('float64').clip(0))
        return df_out


# --- KÜTÜPHANENİN TEKİL FONKSİYONLARININ NASIL KULLANILACAĞINI GÖSTEREN BASİT ÖRNEK ---
if __name__ == '__main__':
    print("### RollingFeaturePipeline Kütüphanesi - Tekil Fonksiyon Gösterimi ###")
    
    # Örnek veri setini, hatanın oluşabileceği bir durumu da içerecek şekilde küçültelim
    sample_df = pd.DataFrame({'Close': np.linspace(100, 150, 50), 'Open': np.linspace(99, 149, 50), 'pct_change': np.random.randn(50)})
    print(f"\nÖrnek veri seti oluşturuldu ({len(sample_df)} satır).")
    
    # 1. Pipeline'ı varsayılan ayarlarla başlat
    pipeline = RollingFeaturePipeline(swt_window_size=16, swt_level=3)
    print("Pipeline varsayılan ayarlarla başlatıldı.")
    
    # 2. 'add_rolling_swt_decomposition' aracını test edelim
    print("\n'add_rolling_swt_decomposition' aracı test ediliyor...")
    df_with_swt = pipeline.add_rolling_swt_decomposition(sample_df, columns_to_process=['Close', 'pct_change'])
    print("Rolling SWT katsayıları eklendi. Sonuç başlığı:")
    # NaN değerlerin oluşup oluşmadığını kontrol edelim
    print("NaN Değer Sayıları:\n", df_with_swt[['Close_cA3', 'Close_cD1', 'pct_change_cA3']].isna().sum())
    print("\nSon Satırlar:")
    print(df_with_swt[['Close', 'Close_cA3', 'Close_cD1', 'Close_cD2', 'Close_cD3', 'Close_cD4']].tail())
    print("\nKütüphane testleri başarılı. Araçlar istisnai durumlara karşı daha dayanıklı.")
