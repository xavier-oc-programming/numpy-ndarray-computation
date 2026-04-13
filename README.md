# NumPy NDArray Computation

Hands-on NumPy exercises covering ndarray creation, manipulation, broadcasting, matrix multiplication, and image processing.

This project explores how NumPy's `ndarray` works as the foundation for numerical computation in Python. The analysis investigates how n-dimensional arrays can be created, inspected, sliced, and operated on — from simple 1D vectors through 3D tensors — and demonstrates how the same mathematical model applies directly to real-world image data. Exercises answer questions such as: how do NumPy arrays differ from Python lists, how does broadcasting enable scalar operations on entire arrays, how does matrix multiplication work in practice, and how can images be represented and transformed as numerical arrays?

The dataset for this project is a custom photograph (`yummy_macarons.jpg`, ~111 KB) used to demonstrate real-world ndarray manipulation. In addition, a raccoon image is loaded directly from SciPy's built-in datasets (`scipy.datasets.face()`). No data cleaning is required — both images are loaded as raw pixel arrays using NumPy's numeric representation, and transformations (greyscale conversion, flipping, rotating, colour inversion) are applied directly to the resulting ndarray values.

No external services or API keys are required. All computation is fully local using NumPy, SciPy, Matplotlib, and Pillow.

---

## Table of Contents

1. [Quick start](#1-quick-start)
2. [Analysis flow](#2-analysis-flow)
3. [Features](#3-features)
4. [Dataset schema](#4-dataset-schema)
5. [Architecture](#5-architecture)
6. [Notebook reference](#6-notebook-reference)
7. [Configuration reference](#7-configuration-reference)
8. [Course context](#8-course-context)
9. [Dependencies](#9-dependencies)

---

## 1. Quick start

```bash
git clone https://github.com/xavier-oc-programming/numpy-ndarray-computation.git
cd numpy-ndarray-computation
pip install -r requirements.txt
jupyter notebook
```

Open `practice/A_01_NumPy_Exercises.ipynb` to run the exercises.  
Open any notebook in `theory/` to read the annotated lesson notes.

---

## 2. Analysis flow

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
└── plt.plot(x, y)              →  plots 1D NumPy vectors as a line chart
```

---

## 3. Features

- Create 1D, 2D, and 3D ndarrays manually and with generator functions
- Slice, reverse, and subset arrays using Python index syntax
- Find non-zero element indices with `np.nonzero()`
- Generate random arrays of arbitrary shape
- Create evenly spaced vectors with `np.arange()` and `np.linspace()`
- Plot NumPy arrays directly with Matplotlib
- Perform element-wise arithmetic on arrays (broadcasting)
- Multiply matrices using `@` operator and `np.matmul()`
- Load a raccoon image from SciPy and inspect its pixel array shape
- Convert an RGB image to greyscale using luminance-weighted dot product
- Flip, rotate, and invert (solarize) images as pure array operations
- Load a custom JPEG with Pillow and invert its colours

---

## 4. Dataset schema

### `data/yummy_macarons.jpg`

Not a tabular dataset — an RGB image loaded as a NumPy ndarray.

| Property | Value | Description |
|---|---|---|
| Shape | `(H, W, 3)` | Height × Width × RGB channels |
| Dtype | `uint8` | Pixel values 0–255 per channel |
| Source | Local photograph | Custom image used to practise ndarray ops |

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

## 5. Architecture

```
numpy-ndarray-computation/
│
├── theory/                                   # Annotated lesson notes
│   ├── 00__Overview.ipynb                    # Day goals and NumPy context
│   ├── 01__NumPy_ndarray.ipynb               # ndarray concept, import setup
│   ├── 02__Generating_Manipulating_ndarrays.ipynb  # arange, linspace, random, slicing
│   ├── 03__Broadcasting_Matrix_Multiplication.ipynb # Vectors, scalars, matmul
│   ├── 04__Images_as_ndarrays.ipynb          # Images as 3D arrays, transformations
│   └── 05__Summary.ipynb                     # Lesson wrap-up and key takeaways
│
├── practice/
│   └── A_01_NumPy_Exercises.ipynb            # Student exercises with solutions
│
├── data/
│   └── yummy_macarons.jpg                    # Custom image for PIL loading exercise
│
├── docs/
│   └── COURSE_NOTES.md                       # Original exercise brief and key concepts
│
├── requirements.txt                          # Pinned package versions
├── .gitignore
└── README.md
```

---

## 6. Notebook reference

### theory/

| Notebook | Key methods covered | Question answered |
|---|---|---|
| `00__Overview.ipynb` | — | What will we build and why does NumPy matter? |
| `01__NumPy_ndarray.ipynb` | `np.array()`, `.shape`, `.ndim`, indexing | What is an ndarray and how is it structured? |
| `02__Generating_Manipulating_ndarrays.ipynb` | `np.arange()`, `np.linspace()`, `np.random.random()`, `np.nonzero()`, `np.flip()`, slicing | How do you generate and reshape arrays efficiently? |
| `03__Broadcasting_Matrix_Multiplication.ipynb` | `+`, `*`, scalar ops, `@`, `np.matmul()` | How does NumPy handle arithmetic across arrays of different shapes? |
| `04__Images_as_ndarrays.ipynb` | `scipy.datasets.face()`, `plt.imshow()`, `np.rot90()`, `np.flip()`, sRGB dot product | How are images represented and manipulated as ndarrays? |
| `05__Summary.ipynb` | — | What were the key takeaways from this lesson? |

### practice/

| Notebook | Key methods covered | Question answered |
|---|---|---|
| `A_01_NumPy_Exercises.ipynb` | Full set: `np.array()`, `np.arange()`, `np.linspace()`, `np.random.random()`, `np.nonzero()`, `np.flip()`, `np.rot90()`, `@`, `np.matmul()`, `Image.open()` | All 12 challenges from the course day — ndarray creation through image inversion |

---

## 7. Configuration reference

| Value | Location | Description |
|---|---|---|
| `"../data/yummy_macarons.jpg"` | `practice/A_01_NumPy_Exercises.ipynb` cell 68 | Relative path from `practice/` to image file |
| `grey_vals = np.array([0.2126, 0.7152, 0.0722])` | `practice/A_01_NumPy_Exercises.ipynb` | ITU-R BT.709 luminance weights for RGB→greyscale |
| `noise = np.random.random((128, 128, 3))` | `practice/A_01_NumPy_Exercises.ipynb` | Fixed shape for the noise image challenge |

---

## 8. Course context

100 Days of Code: The Complete Python Pro Bootcamp — Day 77: Computation with NumPy and N-Dimensional Arrays.  
See [docs/COURSE_NOTES.md](docs/COURSE_NOTES.md) for the full exercise brief and concept summary.

---

## 9. Dependencies

| Module | Used in | Purpose |
|---|---|---|
| `numpy` | All notebooks | ndarray creation, manipulation, math operations |
| `matplotlib` | `theory/02__`, `theory/03__`, `theory/04__`, `practice/A_01__` | Plotting arrays and displaying images |
| `scipy` | `theory/04__`, `practice/A_01__` | Built-in raccoon sample image (`scipy.datasets.face()`) |
| `Pillow` | `theory/04__`, `practice/A_01__` | Loading local JPEG files with `Image.open()` |
| `notebook` | All | Jupyter Notebook server |
