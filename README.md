parthe solid

## Visualisasi state (Batch 1, anggota C)

Setup dari root repository (PowerShell, Python 3.10+):

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
python -m parthevia.visualization_example
python -m unittest discover -s tests -v
```

Contoh menyimpan `output/visualization/state_comparison.png`, `initial_state.json`,
dan `final_state.json`. Ini contoh tetap untuk pengecekan visual, bukan hasil
algoritma atau eksperimen. Initial value 24 dan final value 36; paket B ditumpuk
di atas A dan dirotasi (dimensi asli 1x2x1, efektif 2x1x1). D masuk pada final,
sedangkan E tetap di luar tanpa koordinat. Kedua state diperiksa dengan
`ConstraintChecker` aktual sebelum digambar.

Pemakaian dari modul anggota lain:

```python
import matplotlib.pyplot as plt
from parthevia.visualization import plot_state, plot_state_comparison

fig = plot_state_comparison(result.initial_state, result.final_state,
                            "output/comparison.png")
plt.close(fig)
fig = plot_state(result.final_state, "output/final.png")
plt.close(fig)
```

Kedua fungsi mengembalikan `Figure`, tidak memanggil `show()`, dan tidak mengubah
state. `path` boleh dihilangkan untuk mengatur atau menampilkan figure sendiri.
Perbandingan menggunakan batas sumbu, rasio dimensi, sudut pandang, dan warna ID
yang sama. `package_colors(initial, final)` dapat diteruskan sebagai `colors=`
ke beberapa pemanggilan terpisah. Palet berulang setelah 20 ID; label dan daftar
tetap membedakan paket. Tabel di bawah gambar memuat posisi dan dimensi efektif
paket di dalam serta seluruh paket di luar. Gambar padat mungkin memerlukan
ukuran figure atau sudut pandang lain. Visualisasi tidak memperbaiki atau
memvalidasi state input secara otomatis; gunakan checker sebelum menggambar.

API shared yang dipakai:

- `Package.position`: sudut bawah `(x, y, z)`; `None` berarti di luar.
- `Package.effective_dimensions`: ukuran sepanjang sumbu global `(x, y, z)`
  sesudah orientasi; `orientation` memetakan dimensi lokal w/l/h ke sumbu global.
- `Truck.effective_dimensions`, `width`, `length`, `height`, `max_capacity`:
  ukuran peti kemas dan kapasitas berat.
- `State`: iterable paket, akses `state[id]`, `inside_packages`,
  `outside_packages`, `total_value`, `total_weight`, `copy()`, dan `to_dict()`.
- `ConstraintChecker(truck).check(state)`: `ValidationResult(valid, violations)`;
  `is_valid(state)` tersedia sebagai method maupun fungsi modul model.
- `objective(state)`: total value paket di dalam jika valid, default `-inf`
  jika tidak valid (penalti dapat diganti).
- IO: `load_problem`, `load_state`, `state_from_dict`, `save_state`, `save_json`.

`result.py` menyediakan dataclass `SearchResult` sesuai pemanggilan SA,
Stochastic HC, dan Random Restart HC. `HillClimbingResult` tetap memakai kontrak
semula; kedua bentuk hasil menyediakan `initial_state` dan `final_state` untuk
visualisasi. Serialisasi hasil pencarian tidak ditambahkan pada batch ini;
`params` Random Restart saat ini memuat callable `inner` dari implementasi B.

Implementasi batch ini tidak mengadaptasi kode atau metode dari sumber eksternal.
