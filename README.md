![NumPy](https://img.shields.io/badge/NumPy-blue) ![Matplotlib](https://img.shields.io/badge/Matplotlib-orange) ![SciPy](https://img.shields.io/badge/SciPy-blue) ![Pillow](https://img.shields.io/badge/Pillow-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-orange) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-black)

# NumPy NDArray Computation

Images are just numbers — and NumPy makes that concrete. A 768 × 1024 photograph of a raccoon is a `(768, 1024, 3)` array of integers; converting it to greyscale is a single dot product against three luminance weights; inverting it is `255 - img`. This project works through that idea from first principles, starting with 1D vectors and ending with real pixel transformations on two images.

The analysis covers the full range of ndarray operations: creation and inspection, slicing and indexing, broadcasting, matrix multiplication, and image manipulation. Every transformation — greyscale conversion using the ITU-R BT.709 luminance formula, spatial flips and rotations, colour inversion — is implemented as a direct array operation with no image library doing the heavy lifting.

The custom photograph (`yummy_macarons.jpg`, `533 × 799` px, 3 channels) is colour-inverted using `255 - img_array`. The SciPy raccoon image (`768 × 1024` px) is converted to greyscale via `sRGB_array @ [0.2126, 0.7152, 0.0722]`, flipped with `np.flip()`, rotated with `np.rot90()`, and solarised with `255 - img`. All computation is local — no external services or API keys required.

---

## Quick Start

```bash
git clone https://github.com/xavier-oc-programming/numpy-ndarray-computation.git
cd numpy-ndarray-computation
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/analysis/A_01_NumPy_Exercises.ipynb` to run the analysis.  
Open any notebook in `notebooks/concepts/` to read the annotated reference notes.

---

## Analysis Flow

```
│
│  ── Ingestion ──────────────────────────────────────────────────────────
├── PIL Image.open()            →  loads yummy_macarons.jpg as a PIL Image object
├── scipy.datasets.face()       →  loads built-in raccoon sample image
│
│  ── Array Conversion ───────────────────────────────────────────────────
├── np.array(img)               →  converts PIL / SciPy image to 3D ndarray (H, W, 3)
├── .shape  /  .ndim            →  inspects axis sizes and number of dimensions
│
│  ── Normalisation ──────────────────────────────────────────────────────
├── img / 255                   →  scales pixel values from [0, 255] to [0, 1]  (sRGB)
│
│  ── Greyscale Conversion ───────────────────────────────────────────────
├── sRGB_array @ grey_vals      →  dot product with [0.2126, 0.7152, 0.0722] collapses RGB → luminance
├── plt.imshow(cmap='gray')     →  renders 2D luminance array as a greyscale image
│
│  ── Spatial Transforms ─────────────────────────────────────────────────
├── np.flip(img_gray)           →  reverses the array along axis 0 (flips upside down)
├── np.rot90(img)               →  rotates the colour image 90° counter-clockwise
│
│  ── Pixel Transforms ───────────────────────────────────────────────────
├── 255 - img                   →  inverts every pixel value  (solarize / colour inversion)
│
│  ── Visualisation ──────────────────────────────────────────────────────
├── plt.imshow()                →  displays any ndarray as an image in the notebook
├── plt.plot(x, y)              →  plots 1D NumPy vectors as a line chart
└── plt.savefig()               →  saves all charts to plots/ at 150 dpi
```

---

## Key Findings

- A colour image is a rank-3 tensor of shape `(H, W, 3)` — every pixel is three integers
- Greyscale conversion via luminance dot product collapses the last axis: `(768, 1024, 3)` → `(768, 1024)`
- Broadcasting lets a scalar operation (`255 - img`) transform all 2,359,296 pixel values simultaneously
- Matrix multiplication with `@` produces a `(4, 3)` result from `(4, 2) @ (2, 3)` in one line
- `np.linspace(0, 100, 9)` produces `[0, 12.5, 25, …, 100]` — 9 evenly spaced floats including both endpoints
- The macaron photograph is `533 × 799` px (`426,267` pixels total); the raccoon image is `768 × 1024` px (`786,432` pixels)

---

## Dataset Schema

### `data/yummy_macarons.jpg`

Not a tabular dataset — an RGB image loaded as a NumPy ndarray.

| Property | Value | Description |
|---|---|---|
| Shape | `(533, 799, 3)` | Height × Width × RGB channels |
| Dtype | `uint8` | Pixel values 0–255 per channel |
| Source | Local photograph | Custom image used to demonstrate ndarray ops |

### SciPy raccoon image (`scipy.datasets.face()`)

| Property | Value | Description |
|---|---|---|
| Shape | `(768, 1024, 3)` | 768px tall, 1024px wide, RGB |
| Dtype | `uint8` | Pixel values 0–255 per channel |
| Source | `scipy.datasets` | Built-in SciPy sample image |

**Computed arrays added at runtime:**

| Name | Shape | Description |
|---|---|---|
| `sRGB_array` | `(768, 1024, 3)` | img / 255, values normalised to [0, 1] |
| `img_gray` | `(768, 1024)` | Greyscale via luminance dot product |
| `solar_img` | `(768, 1024, 3)` | Colour-inverted raccoon image |
| `noise` | `(128, 128, 3)` | Random float array displayed as image |

---

## Architecture

```
numpy-ndarray-computation/
│
├── notebooks/
│   ├── analysis/
│   │   └── A_01_NumPy_Exercises.ipynb    # Main analysis notebook
│   └── concepts/
│       ├── 00__Overview.ipynb            # NumPy context and scope
│       ├── 01__NumPy_ndarray.ipynb       # ndarray concept, import setup
│       ├── 02__Generating_Manipulating_ndarrays.ipynb  # arange, linspace, random, slicing
│       ├── 03__Broadcasting_Matrix_Multiplication.ipynb # Vectors, scalars, matmul
│       ├── 04__Images_as_ndarrays.ipynb  # Images as 3D arrays, transformations
│       └── 05__Summary.ipynb             # Key takeaways
│
├── data/
│   └── yummy_macarons.jpg                # Custom image for PIL loading
│
├── plots/                                # All saved charts (150 dpi)
│
├── notebook_web_render/
│   └── index.html                        # Rendered notebook (GitHub Pages)
│
├── docs/
│   └── COURSE_NOTES.md                   # Reference notes
│
├── .github/workflows/
│   └── publish_notebook.yml              # CI/CD: render and deploy on push
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Visualisations

All charts are saved to `plots/` at 150 dpi.

| File | Description |
|---|---|
| `plots/line_chart.png` | `x` vs `y` — linspace vectors plotted as a line chart |
| `plots/noise_image.png` | 128 × 128 × 3 random float array displayed as colour noise |
| `plots/raccoon_original.png` | SciPy raccoon image in colour |
| `plots/raccoon_greyscale.png` | Raccoon converted to greyscale via luminance dot product |
| `plots/raccoon_flipped.png` | Greyscale raccoon flipped upside down with `np.flip()` |
| `plots/raccoon_rotated.png` | Colour raccoon rotated 90° counter-clockwise with `np.rot90()` |
| `plots/raccoon_solarised.png` | Colour raccoon inverted with `255 - img` |
| `plots/macarons_original.png` | Custom macaron photograph loaded with Pillow |
| `plots/macarons_inverted.png` | Macaron image colour-inverted with `255 - img_array` |

---

## Operations Reference

| Value | Location | Description |
|---|---|---|
| `"../../data/yummy_macarons.jpg"` | `notebooks/analysis/A_01_NumPy_Exercises.ipynb` | Relative path from `notebooks/analysis/` to image file |
| `grey_vals = np.array([0.2126, 0.7152, 0.0722])` | `notebooks/analysis/A_01_NumPy_Exercises.ipynb` | ITU-R BT.709 luminance weights for RGB→greyscale |
| `noise = np.random.random((128, 128, 3))` | `notebooks/analysis/A_01_NumPy_Exercises.ipynb` | Fixed shape for the noise image |

---

## Background

100 Days of Code: The Complete Python Pro Bootcamp — Day 77: Computation with NumPy and N-Dimensional Arrays.  
See [docs/COURSE_NOTES.md](docs/COURSE_NOTES.md) for the full exercise brief and concept summary.

---

## Dependencies

| Module | Used in | Purpose |
|---|---|---|
| `numpy` | All notebooks | ndarray creation, manipulation, math operations |
| `matplotlib` | `notebooks/concepts/02__`, `03__`, `04__`, `notebooks/analysis/` | Plotting arrays and displaying images |
| `scipy` | `notebooks/concepts/04__`, `notebooks/analysis/` | Built-in raccoon sample image (`scipy.datasets.face()`) |
| `Pillow` | `notebooks/concepts/04__`, `notebooks/analysis/` | Loading local JPEG files with `Image.open()` |
| `notebook` | All | Jupyter Notebook server |
