# wavw_predict

高精度ハイブリッド津波解析モデル。浅水方程式（SWE）物理ソルバーと深層学習（Fourier Neural Operator）を組み合わせ、建物波力・盛土崩壊・背後伝達を予測する。

---

## 必要環境

- Python 3.10 以上
- PyTorch 2.1 以上
- CUDA（オプション、CPU でも動作）

## インストール

```bash
git clone https://github.com/Arupaka610/wavw_predict.git
cd wavw_predict
pip install -e .
```

---

## データ準備

### シミュレーション結果（COMCOT / GeoClaw）の前処理

```bash
python scripts/preprocess_simulations.py \
    --input-dir data/raw/simulations \
    --bathy-dir data/raw/bathymetry \
    --output-dir data/processed/train \
    --grid-shape 256 256
```

入力ファイル形式:
- `data/raw/simulations/*.nc` — 波動場 NetCDF（変数: `eta`, `u`, `v`、形状 `[T, H, W]`）
- `data/raw/bathymetry/*.nc` — 海底地形 NetCDF（変数: `depth`, `lon`, `lat`）

### 観測データ（潮位計・GPS）の前処理

```bash
python scripts/preprocess_observations.py \
    --gauge-dir data/raw/observations/event_001 \
    --output-dir data/processed/observations \
    --dt 60
```

入力ファイル形式:
- `*.csv` — 各潮位計 1 ファイル、列 `time`（datetime）と `eta`（m）必須
- ファイル名規則: `gauge_<lat>_<lon>.csv`

---

## モデルの学習

### 1. FNO サロゲートモデル（波動伝播）

```bash
python -m wavw_predict.training.train_surrogate \
    --config configs/train_surrogate.yaml
```

### 2. 波力係数ネットワーク（建物波力）

```bash
python -m wavw_predict.training.train_wave_force \
    --config configs/train_wave_force.yaml
```

### 3. 盛土崩壊ネットワーク

```bash
python -m wavw_predict.training.train_embankment \
    --config configs/train_embankment.yaml
```

チェックポイントは `checkpoints/` ディレクトリに保存される。

---

## 推論（本番実行）

```bash
python scripts/run_inference.py \
    --config configs/base.yaml \
    --ic-file data/processed/ic_example.npz \
    --fno-ckpt checkpoints/fno/best.pt \
    --wf-ckpt checkpoints/wave_force/best.pt \
    --emb-ckpt checkpoints/embankment/best.pt \
    --n-steps 1800 \
    --n-forecast 600 \
    --output results/run_001.npz \
    --device cuda
```

初期条件ファイル（`.npz`）が含むべき変数:

| 変数 | 形状 | 説明 |
|------|------|------|
| `eta` | `[H, W]` | 初期水面変位 [m] |
| `u` | `[H, W]` | 初期 x 方向流速（省略可） |
| `v` | `[H, W]` | 初期 y 方向流速（省略可） |
| `bathy` | `[H, W]` | 海底地形（正=水深）[m] |
| `mask` | `[H, W]` | 陸域マスク（1=陸地） |

出力ファイル（`.npz`）:

| 変数 | 説明 |
|------|------|
| `eta` | 波動場 `[T, H, W]` |
| `u`, `v` | 流速場 |
| `force_<name>` | 各構造物の波力 [N/m] |
| `force_behind_<name>` | 背後伝達波力 [N/m] |

---

## Python API からの使用

### 物理ソルバー単体

```python
import torch
from wavw_predict.physics.swe_solver import SWESolver2D

nx, ny = 256, 256
bathymetry = torch.full((ny, nx), 10.0)   # 水深 10m の平坦地形

solver = SWESolver2D(
    dx=100.0, dy=100.0, dt=2.0,
    nx=nx, ny=ny,
    bathymetry=bathymetry,
)

eta0 = torch.zeros(ny, nx)
u0 = torch.zeros(ny, nx)
v0 = torch.zeros(ny, nx)

eta_hist, u_hist, v_hist = solver.run(eta0, u0, v0, n_steps=300)
# eta_hist.shape → [301, 256, 256]
```

### ハイブリッドパイプライン

```python
from wavw_predict.hybrid.pipeline import (
    HybridPipeline, PipelineConfig, InitialConditions, StructureSpec,
)
from wavw_predict.physics.wave_force import StructureGeometry
import torch

cfg = PipelineConfig(dx=100.0, dy=100.0, dt=2.0, device="cpu")
pipeline = HybridPipeline.from_checkpoints(
    cfg,
    fno_ckpt="checkpoints/fno/best.pt",
    wf_ckpt="checkpoints/wave_force/best.pt",
    emb_ckpt="checkpoints/embankment/best.pt",
)

ny, nx = 128, 128
ic = InitialConditions(
    eta=torch.zeros(ny, nx),
    u=torch.zeros(ny, nx),
    v=torch.zeros(ny, nx),
    bathymetry=torch.full((ny, nx), 8.0),
    coast_mask=torch.zeros(ny, nx),
)

structures = [
    StructureSpec(
        name="building_A",
        grid_row=64, grid_col=80,
        geometry=StructureGeometry(width=10.0, height=8.0, length=15.0),
    ),
]

result = pipeline.run(ic, n_steps=900, structures=structures)

print(result.eta.shape)                         # [901, 128, 128]
print(result.building_forces["building_A"])     # [901] 波力時系列 [N/m]
```

### 波力の単体計算

```python
from wavw_predict.physics.wave_force import MorisonForceCalculator, StructureGeometry
import torch

calc = MorisonForceCalculator(rho=1025.0, g=9.81, Cd=2.0, Cm=2.0)
geom = StructureGeometry(width=10.0, height=8.0, length=15.0)

h = torch.tensor([4.0])    # 水深 [m]
u = torch.tensor([3.0])    # 流速 [m/s]
du_dt = torch.tensor([0.5])

result = calc.compute(h, u, du_dt, geom)
print(f"波力合計: {result.F_total.item():.1f} N/m")
print(f"抗力:     {result.F_drag.item():.1f} N/m")
print(f"慣性力:   {result.F_inertia.item():.1f} N/m")
```

### 盛土崩壊の単体シミュレーション

```python
from wavw_predict.physics.embankment import EmbankmentModel, EmbankmentProperties

props = EmbankmentProperties(
    crest_elevation=3.0,    # 天端標高 [m]
    crest_width=5.0,        # 天端幅 [m]
    upstream_slope=2.0,     # 上流側勾配 (H:V)
    downstream_slope=2.0,
    D50=0.001,              # 中央粒径 [m]
    tau_critical=1.5,       # 限界掃流力 [Pa]
)
model = EmbankmentModel(props)

dt = 1.0
for t in range(600):
    H_upstream = 4.0 + 0.5 * (t / 600)   # 上流水位が徐々に上昇
    state = model.step(H_upstream, dt)
    if state.is_failed:
        print(f"崩壊発生 t={t}s  越流幅={state.breach_width:.1f}m")
        break
```

---

## テスト実行

```bash
python -m pytest tests/ -v
```

テスト内容（26 件）:

| ファイル | テスト内容 |
|----------|-----------|
| `test_swe_solver.py` | 静水安定・質量保存・波速 |
| `test_wave_force.py` | Morison 式・静水圧・伝達係数 |
| `test_embankment.py` | 越流流量・崩壊トリガー・リセット |
| `test_models.py` | FNO/WaveForceNet/EmbankmentNet 出力形状・物理境界 |
| `test_pipeline.py` | パイプライン統合・有限値チェック |

---

## 設定ファイル

`configs/base.yaml` で主要パラメータを変更できる:

```yaml
grid:
  dx: 100.0     # 空間解像度 x [m]
  dy: 100.0     # 空間解像度 y [m]
  dt: 2.0       # タイムステップ [s]
  nx: 256       # グリッド数 x
  ny: 256       # グリッド数 y

training:
  fno:
    lr: 1.0e-3
    batch_size: 16
    epochs: 100
```

---

## アーキテクチャ

```
入力: 地震パラメータ / 初期条件 / 観測データ
  │
  ├─ SWE ソルバー (2D Lax-Wendroff)
  │    └─ 低解像度波動場 [T, H, W]
  │
  ├─ FNO サロゲート (Fourier Neural Operator)
  │    └─ 高解像度精緻化 + 予測延長
  │
  ├─ WaveForceNet (MLP, 14次元特徴量)
  │    └─ Cd/Cm/Kt 係数 → Morison 式 → 建物波力
  │
  ├─ EmbankmentNet (拡張因果 TCN)
  │    └─ 崩壊確率 p_failure + 崩壊幅/深さ (ガウス分布)
  │
  └─ 出力:
       建物波力 F [N/m]
       背後波力 F_behind [N/m]
       崩壊確率 p_failure [0,1]
       崩壊規模 breach_width/depth [m]
```
