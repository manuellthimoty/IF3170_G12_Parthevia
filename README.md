# Parthevia

## Setup

```bash
pip install -r requirements.txt
```

## Cara run eksperimen

Dari root repo, jalankan satu algoritma sebanyak 3 kali:

PowerShell:
```powershell
$env:PYTHONPATH = "src"
python -m parthevia.experiment sa data/sample_medium.json
```

Git Bash / Linux / macOS:
```bash
PYTHONPATH=src python -m parthevia.experiment sa data/sample_medium.json
```

Nama algoritma (lihat `ALGORITHMS` di `src/parthevia/algorithms/__init__.py`):
`sa`, `stochastic-hc`, `random-restart-hc`, `steepest-ascent-hc`, `sideways-hc`, `ga`

Mengubah jumlah run, seed, atau parameter algoritma (dari Python):

```python
from parthevia.experiment import run_experiment

run_experiment("sa", "data/sample_medium.json", runs=3, seed=0, cooling_rate=0.995)
run_experiment("random-restart-hc", "data/sample_medium.json", max_restart=10)
run_experiment("sideways-hc", "data/sample_medium.json", max_sideways=20)
```

Eksperimen GA (18 run: 3 variasi iterasi dan 3 variasi populasi) dijalankan lewat `ga_experiments.py`:

```bash
PYTHONPATH=src python -m parthevia.ga_experiments --input data/ga_synthetic.json --output output/<folder_baru>
```

Folder `--output` harus belum ada (hasil lama tidak akan ditimpa). Opsi lain: `--mode verify` untuk mengaudit hasil, `--mode plot` untuk membuat ulang grafik, `--objective` untuk memilih fungsi objektif.

## Hasil

Hasil tiap run `i` disimpan di `output/<algoritma>/`:

| File | Isi |
|---|---|
| `run_i.json` | Hasil lengkap: state awal & akhir, OF, iterasi, histori OF, durasi |
| `run_i/objective.png` | Plot OF vs iterasi |
| `run_i/states.png` | Visualisasi 3D state awal vs akhir |
| `run_i/sa_exponential.png` | Plot e^(ΔE/T) vs iterasi (khusus SA) |

Durasi ada di `extra["duration_seconds"]` (hasil `SearchResult`) atau `durationSeconds`