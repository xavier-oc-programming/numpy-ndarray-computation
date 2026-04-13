# Course Notes — Day 77: Computation with NumPy and N-Dimensional Arrays

**Course**: 100 Days of Code: The Complete Python Pro Bootcamp  
**Day**: 77  
**Topics**: NumPy ndarray, array generation, broadcasting, matrix multiplication, image manipulation

---

## Exercise Brief

No data science course is complete without **NumPy (Numerical Python)**. NumPy is a foundational Python library used across science, engineering, machine learning, data analysis, and computer vision. It is the standard for numerical computation in Python, and many other libraries (Pandas, SciPy, Matplotlib) are built on top of it.

This lesson shifts focus from high-level data manipulation (Pandas) to low-level numerical computation.

---

## Key Concepts Covered

### 1. The ndarray

The core NumPy object is the **ndarray** — a homogeneous n-dimensional array.

- **Homogeneous**: all elements share the same data type
- **N-dimensional**: supports 1D vectors, 2D matrices, and higher-order tensors
- Much faster than Python lists due to contiguous memory and vectorised C operations

### 2. Creating Arrays

| Function | Description |
|---|---|
| `np.array([...])` | Create from a Python list |
| `np.arange(start, stop)` | Evenly spaced integers |
| `np.linspace(start, stop, num)` | Evenly spaced floats, endpoints included |
| `np.random.random(shape)` | Random floats in [0, 1) |

### 3. Inspecting Arrays

- `.shape` — tuple of axis sizes, e.g. `(3, 4)` for a 3×4 matrix
- `.ndim` — number of dimensions
- Indexing: `arr[row, col]` for 2D, `arr[i, j, k]` for 3D

### 4. Slicing

Same as Python list slicing but works on each axis:

```python
arr[2:]        # from index 2 to end
arr[3:6]       # indices 3, 4, 5
arr[::2]       # every second element
arr[::-1]      # reverse
```

### 5. Broadcasting

NumPy applies scalar operations element-wise across the entire array:

```python
v * 2          # doubles every element
array_2d + 10  # adds 10 to every element
```

Python lists do not support this — `list * 2` repeats the list, not its values.

### 6. Linear Algebra

- Element-wise addition/multiplication: `v1 + v2`, `v1 * v2`
- Matrix multiplication (dot product): `a @ b` or `np.matmul(a, b)`
- Shape rule: `(m×n) @ (n×p) → (m×p)`

### 7. Images as ndarrays

An image is a 3D ndarray with shape `(height, width, 3)` where 3 = RGB channels.

```python
img = datasets.face()    # scipy sample raccoon image
img.shape                # (768, 1024, 3)
```

Operations:
- **Greyscale**: multiply by `[0.2126, 0.7152, 0.0722]` (luminance weights)
- **Flip**: `np.flip(arr)`
- **Rotate**: `np.rot90(arr)`
- **Invert / solarize**: `255 - arr`

---

## Challenges Completed

1. Generate a range vector with `np.arange()`
2. Slice subsets using Python slicing syntax
3. Reverse an array with `[::-1]` and `np.flip()`
4. Find non-zero indices with `np.nonzero()`
5. Generate a 3×3×3 random array
6. Create evenly spaced vectors with `np.linspace()`
7. Plot `x` vs `y` with Matplotlib
8. Generate a 128×128×3 noise image and display it
9. Load a raccoon image from SciPy and inspect its shape
10. Convert to greyscale using sRGB luminance weights
11. Flip, rotate, and invert (solarize) images
12. Load a custom image (yummy_macarons.jpg) with PIL and invert its colours
