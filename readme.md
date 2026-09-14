<div align="center">

<br />

```
 ██████╗  █████╗ ████████╗ █████╗     ███████╗ ██████╗██╗███████╗███╗   ██╗ ██████╗███████╗
 ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██╔════╝██╔════╝██║██╔════╝████╗  ██║██╔════╝██╔════╝
 ██║  ██║███████║   ██║   ███████║    ███████╗██║     ██║█████╗  ██╔██╗ ██║██║     █████╗  
 ██║  ██║██╔══██║   ██║   ██╔══██║    ╚════██║██║     ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝  
 ██████╔╝██║  ██║   ██║   ██║  ██║    ███████║╚██████╗██║███████╗██║ ╚████║╚██████╗███████╗
 ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
```

### ✦ The Complete Python Data Science Mastery Repository ✦
### NumPy · Pandas · Matplotlib · Seaborn · SciPy

<br />

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-4c72b0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)
[![License](https://hila-11.com)



<br />

<a href="#-overview">Overview</a> &nbsp;•&nbsp;
<a href="#-libraries">Libraries</a> &nbsp;•&nbsp;
<a href="#-project-structure">Structure</a> &nbsp;•&nbsp;
<a href="#-getting-started">Getting Started</a> &nbsp;•&nbsp;
<a href="#-notebooks">Notebooks</a> &nbsp;•&nbsp;
<a href="#-cheatsheets">Cheatsheets</a>

<br />

---

<br />

> **"Data is the new oil — but only if you know how to refine it."**
> This repo is your complete refinery.

<br />

</div>

---

## 📌 Overview

This repository is a **comprehensive, hands-on mastery guide** to the five core Python data science libraries — from the ground up to advanced production-level usage. Whether you're an absolute beginner stepping into data science or an experienced developer looking to sharpen your analytical toolkit, this repo covers everything with clean code, visual examples, and real-world datasets.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│   NUMPY          →   Numerical computation, arrays, linear algebra           │
│     ↓                                                                         │
│   PANDAS         →   Data wrangling, cleaning, analysis, aggregation         │
│     ↓                                                                         │
│   MATPLOTLIB     →   Core plotting, figures, axes, customization             │
│     ↓                                                                         │
│   SEABORN        →   Statistical visualization, beautiful charts             │
│     ↓                                                                         │
│   SCIPY          →   Scientific computing, stats, optimization, signal       │
│                                                                               │
│         Raw Data  →  Clean Data  →  Analysis  →  Visualization  →  Insight   │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

<br />

---

## 📚 Libraries — Deep Dive

<br />

### `01` &nbsp; 🔢 &nbsp; NumPy — Numerical Python

> The backbone of all scientific computing in Python. NumPy provides the `ndarray` — a blazing-fast multidimensional array object that powers nearly every data science library in the ecosystem.

**What you'll master:**

| Topic | Description |
|---|---|
| 🏗️ Array Creation | `zeros`, `ones`, `eye`, `arange`, `linspace`, `random`, `full` |
| 📐 Array Indexing | Basic, boolean, fancy indexing, slicing across N dimensions |
| 🔄 Reshaping & Manipulation | `reshape`, `transpose`, `flatten`, `ravel`, `squeeze`, `expand_dims` |
| ➕ Arithmetic & Broadcasting | Element-wise ops, broadcasting rules, vectorized computations |
| 📊 Aggregations | `sum`, `mean`, `std`, `var`, `min`, `max`, `cumsum`, `prod` along axes |
| 🧮 Linear Algebra | `dot`, `matmul`, `inv`, `det`, `eig`, `svd`, `solve` via `np.linalg` |
| 🎲 Random Module | Distributions: normal, uniform, binomial, Poisson, seeding for reproducibility |
| 💾 File I/O | `np.save`, `np.load`, `np.savetxt`, `np.genfromtxt` |
| ⚡ Performance | Vectorization vs loops, memory layout (C vs Fortran order), `np.vectorize` |

**Quick taste:**

```python
import numpy as np

# Create a 3D array and perform advanced operations
arr = np.random.randn(100, 50, 3)

# Broadcasting — no loops needed
normalized = (arr - arr.mean(axis=0)) / arr.std(axis=0)

# Fancy indexing
mask = arr[:, :, 0] > 0.5
selected = arr[mask]

# Linear algebra
A = np.random.rand(4, 4)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Matrix rank: {np.linalg.matrix_rank(A)}")
print(f"Determinant: {np.linalg.det(A):.4f}")
```

<br />

---

### `02` &nbsp; 🐼 &nbsp; Pandas — Data Analysis Library

> The most powerful data manipulation library in Python. Pandas brings the intuition of spreadsheets and SQL into Python with its `DataFrame` and `Series` structures — but supercharged.

**What you'll master:**

| Topic | Description |
|---|---|
| 📂 Data Loading | CSV, Excel, JSON, SQL, Parquet, HTML, clipboard — `read_*` functions |
| 🔍 Exploration | `head`, `tail`, `info`, `describe`, `dtypes`, `shape`, `value_counts` |
| 🧹 Data Cleaning | Handling nulls, duplicates, type casting, string cleaning, renaming |
| 🎯 Indexing | `loc`, `iloc`, `at`, `iat`, boolean indexing, `query()`, MultiIndex |
| 🔀 Reshaping | `pivot`, `pivot_table`, `melt`, `stack`, `unstack`, `crosstab` |
| 🔗 Merging & Joining | `merge`, `join`, `concat`, `combine_first` — SQL-style operations |
| 📊 GroupBy | Split-apply-combine, `agg`, `transform`, `apply`, `filter`, `pipe` |
| 📅 Time Series | `DatetimeIndex`, resampling, rolling, shifting, time zone handling |
| ⚡ Performance | `eval`, `query`, chunking large files, categorical dtypes, `pyarrow` backend |
| 📤 Exporting | `to_csv`, `to_excel`, `to_json`, `to_parquet`, `to_sql` |

**Quick taste:**

```python
import pandas as pd

# Load and explore
df = pd.read_csv("sales_data.csv", parse_dates=["date"])
print(df.info())
print(df.describe())

# Clean
df = df.dropna(subset=["revenue"])
df["category"] = df["category"].str.strip().str.lower()
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

# Analyze — GroupBy + Aggregation
summary = (
    df.groupby(["region", "category"])
    .agg(
        total_revenue=("revenue", "sum"),
        avg_revenue=("revenue", "mean"),
        order_count=("order_id", "nunique"),
    )
    .sort_values("total_revenue", ascending=False)
    .reset_index()
)

# Time Series — monthly rolling average
df = df.set_index("date").sort_index()
df["rolling_avg"] = df["revenue"].rolling("30D").mean()
monthly = df["revenue"].resample("ME").sum()
```

<br />

---

### `03` &nbsp; 📊 &nbsp; Matplotlib — Core Visualization

> The foundational plotting library of Python. Matplotlib gives you complete control over every pixel of your figures — from simple line charts to publication-quality scientific plots.

**What you'll master:**

| Topic | Description |
|---|---|
| 🎨 Figure & Axes API | `plt.figure`, `plt.subplots`, `fig.add_subplot`, OOP vs pyplot interface |
| 📈 Plot Types | Line, bar, scatter, histogram, pie, area, step, stem, boxplot, violin |
| 🧭 Annotations | `annotate`, `text`, `axhline`, `axvline`, `arrow`, `patches` |
| 🎨 Styling | Colormaps, line styles, markers, color cycles, `rcParams`, stylesheets |
| 🔲 Layouts | `GridSpec`, `subplot_mosaic`, `tight_layout`, `constrained_layout` |
| 💾 Saving | `savefig` with DPI, format (SVG, PDF, PNG), transparent backgrounds |
| 🔗 3D Plotting | `Axes3D`, surface plots, wireframe, 3D scatter, contour |
| 🎭 Animations | `FuncAnimation`, `ArtistAnimation` for animated visualizations |
| 🖼️ Image Display | `imshow`, colormaps, normalization for raster data |

**Quick taste:**

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(16, 10), facecolor="#0f0f0f")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1 — Time series with shading
ax1 = fig.add_subplot(gs[0, :2])
x = np.linspace(0, 4 * np.pi, 500)
y = np.sin(x) * np.exp(-0.1 * x)
ax1.plot(x, y, color="#61DAFB", linewidth=2, label="Signal")
ax1.fill_between(x, y, alpha=0.15, color="#61DAFB")
ax1.set_facecolor("#1a1a2e")
ax1.set_title("Damped Sine Wave", color="white", fontsize=13)

# Panel 2 — Scatter with colormap
ax2 = fig.add_subplot(gs[0, 2])
n = 300
scatter = ax2.scatter(
    np.random.randn(n), np.random.randn(n),
    c=np.random.randn(n), cmap="plasma", alpha=0.8, s=30
)
fig.colorbar(scatter, ax=ax2)

plt.savefig("output.png", dpi=150, bbox_inches="tight")
plt.show()
```

<br />

---

### `04` &nbsp; 🌊 &nbsp; Seaborn — Statistical Visualization

> Built on top of Matplotlib, Seaborn brings statistical intelligence into your charts. With minimal code, it produces beautiful, informative visualizations that would take dozens of lines in raw Matplotlib.

**What you'll master:**

| Topic | Description |
|---|---|
| 📊 Relational Plots | `scatterplot`, `lineplot` — with hue, size, style dimensions |
| 📦 Distribution Plots | `histplot`, `kdeplot`, `ecdfplot`, `rugplot` |
| 🗂️ Categorical Plots | `barplot`, `boxplot`, `violinplot`, `stripplot`, `swarmplot`, `pointplot` |
| 🔥 Matrix / Heatmaps | `heatmap`, `clustermap` — correlation matrices, confusion matrices |
| 🔗 Regression Plots | `regplot`, `lmplot`, `residplot` — linear and polynomial regression lines |
| 🌐 Multi-plot Grids | `FacetGrid`, `PairGrid`, `pairplot` — variable-by-variable exploration |
| 🎨 Themes & Palettes | `set_theme`, `set_style`, `set_palette`, custom color palettes |
| 📐 Statistical Estimates | Confidence intervals, bootstrap, multiple comparisons built-in |

**Quick taste:**

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="darkgrid", palette="deep", font_scale=1.2)

# Load built-in dataset
tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Categorical — Violin + Strip combo
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex",
               split=True, inner="quart", ax=axes[0])

# Relational — Scatter with regression
sns.regplot(data=tips, x="total_bill", y="tip",
            scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}, ax=axes[1])

# Pairwise — full exploration
sns.pairplot(penguins, hue="species", diag_kind="kde",
             plot_kws={"alpha": 0.6})

# Heatmap — Correlation matrix
corr = tips.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5)

plt.tight_layout()
plt.show()
```

<br />

---

### `05` &nbsp; 🔬 &nbsp; SciPy — Scientific Computing

> SciPy extends NumPy into advanced scientific computing — statistics, optimization, signal processing, linear algebra, interpolation, integration, and more. It's the bridge between data science and engineering/science.

**What you'll master:**

| Module | Description |
|---|---|
| `scipy.stats` | 100+ probability distributions, hypothesis tests (t-test, ANOVA, chi-square, KS test), descriptive stats |
| `scipy.optimize` | Curve fitting, root finding, linear programming, minimization (`minimize`, `curve_fit`, `brentq`) |
| `scipy.interpolate` | 1D/2D interpolation, splines, `interp1d`, `griddata`, `RectBivariateSpline` |
| `scipy.integrate` | Numerical integration, ODEs — `quad`, `dblquad`, `solve_ivp` |
| `scipy.signal` | Filtering, convolution, FFT, spectral analysis, peak finding |
| `scipy.linalg` | Advanced linear algebra — LU, QR, Cholesky, SVD decompositions |
| `scipy.cluster` | Hierarchical clustering, k-means, distance metrics |
| `scipy.spatial` | KD-trees, Voronoi diagrams, Delaunay triangulation, convex hulls |
| `scipy.ndimage` | N-dimensional image processing, morphology, filters |
| `scipy.fft` | Fast Fourier Transform and inverse, frequency analysis |

**Quick taste:**

```python
from scipy import stats, optimize, signal, interpolate
import numpy as np

# ── Hypothesis Testing ──────────────────────────────────────────────────
group_a = np.random.normal(50, 10, 100)
group_b = np.random.normal(53, 10, 100)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
print(f"Significant: {p_value < 0.05}")

# ── Distribution Fitting ─────────────────────────────────────────────────
data = np.random.exponential(scale=2, size=500)
params = stats.expon.fit(data)
ks_stat, ks_p = stats.kstest(data, "expon", args=params)
print(f"KS test p-value: {ks_p:.4f}")

# ── Curve Fitting ─────────────────────────────────────────────────────────
def model(x, a, b, c):
    return a * np.exp(-b * x) + c

x_data = np.linspace(0, 10, 100)
y_data = model(x_data, 3.5, 0.4, 1.2) + np.random.normal(0, 0.2, 100)
popt, pcov = optimize.curve_fit(model, x_data, y_data, p0=[3, 0.5, 1])
print(f"Fitted params: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}")

# ── Signal Processing ─────────────────────────────────────────────────────
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs)
noisy_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(fs)

b, a = signal.butter(4, 100, fs=fs, btype="low")
filtered = signal.filtfilt(b, a, noisy_signal)

# ── Interpolation ─────────────────────────────────────────────────────────
x_sparse = np.array([0, 1, 3, 5, 7, 10])
y_sparse = np.sin(x_sparse)
f_cubic = interpolate.interp1d(x_sparse, y_sparse, kind="cubic")
x_dense = np.linspace(0, 10, 300)
y_dense = f_cubic(x_dense)
```

<br />

---

## 🚀 Getting Started

<br />

### Prerequisites

```
Python     3.10 or higher
pip        23.0+
git        any recent version
```

<br />

### Installation — Option A: pip (Standard)

```bash
# Clone the repository
git clone https://github.com/your-username/numpy-pandas-matplotlib-seaborn-scipy.git
cd numpy-pandas-matplotlib-seaborn-scipy

# Create a virtual environment (strongly recommended)
python -m venv venv

# Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

<br />

### Installation — Option B: conda (Recommended for Data Science)

```bash
# Create a conda environment from the provided file
conda env create -f environment.yml

# Activate it
conda activate datascience

# Launch Jupyter
jupyter notebook
```

<br />

### Installation — Option C: uv (Fastest)

```bash
# Install uv if you haven't already
pip install uv

# Create environment and install in one shot
uv venv && uv pip install -r requirements.txt
```

<br />

### requirements.txt

```txt
numpy>=1.26.0
pandas>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.11.0
jupyter>=1.0.0
jupyterlab>=4.0.0
ipykernel>=6.0.0
notebook>=7.0.0
openpyxl>=3.1.0        # Excel support for Pandas
pyarrow>=14.0.0        # Parquet support
plotly>=5.18.0         # Optional: interactive plots
scikit-learn>=1.3.0    # Optional: ML integration examples
statsmodels>=0.14.0    # Optional: advanced stats
```

<br />

### Launch Jupyter

```bash
jupyter notebook
# or use JupyterLab (recommended)
jupyter lab
```

<br />

---

## 🗃️ Project Structure

```
numpy-pandas-matplotlib-seaborn-scipy/
│
├── 📁 01_numpy/
│   ├── 01_array_basics.ipynb             # ndarray creation, types, attributes
│   ├── 02_indexing_and_slicing.ipynb     # All indexing patterns
│   ├── 03_operations_broadcasting.ipynb  # Arithmetic, broadcasting rules
│   ├── 04_linear_algebra.ipynb           # np.linalg — full coverage
│   ├── 05_random_module.ipynb            # Random distributions & seeding
│   ├── 06_performance_tricks.ipynb       # Vectorization, memory, profiling
│   └── 07_real_world_numpy.ipynb         # Image arrays, signal arrays, tabular
│
├── 📁 02_pandas/
│   ├── 01_series_and_dataframe.ipynb     # Core data structures
│   ├── 02_loading_and_exporting.ipynb    # read_csv, read_excel, JSON, SQL, Parquet
│   ├── 03_data_cleaning.ipynb            # Nulls, duplicates, types, strings
│   ├── 04_indexing_and_filtering.ipynb   # loc, iloc, query, boolean masks
│   ├── 05_groupby_and_agg.ipynb          # Split-apply-combine mastery
│   ├── 06_merge_join_concat.ipynb        # SQL-style joins and concatenation
│   ├── 07_reshaping.ipynb                # pivot, melt, stack, unstack
│   ├── 08_time_series.ipynb              # DatetimeIndex, resample, rolling
│   ├── 09_apply_and_pipe.ipynb           # Custom functions & method chaining
│   └── 10_pandas_performance.ipynb       # eval, query, chunking, categoricals
│
├── 📁 03_matplotlib/
│   ├── 01_figure_and_axes.ipynb          # OOP interface, fig/ax architecture
│   ├── 02_plot_types.ipynb               # Line, bar, scatter, hist, pie, box
│   ├── 03_subplots_and_layouts.ipynb     # GridSpec, subplot_mosaic, tight_layout
│   ├── 04_styling_and_themes.ipynb       # rcParams, stylesheets, colormaps
│   ├── 05_annotations.ipynb              # Text, arrows, patches, shapes
│   ├── 06_3d_plots.ipynb                 # Axes3D, surface, wireframe, contour
│   ├── 07_animations.ipynb               # FuncAnimation, ArtistAnimation
│   └── 08_publication_quality.ipynb      # DPI, fonts, SVG/PDF export
│
├── 📁 04_seaborn/
│   ├── 01_themes_and_palettes.ipynb      # set_theme, styles, color palettes
│   ├── 02_relational_plots.ipynb         # scatterplot, lineplot with dimensions
│   ├── 03_distribution_plots.ipynb       # histplot, kdeplot, ecdfplot
│   ├── 04_categorical_plots.ipynb        # boxplot, violin, bar, strip, swarm
│   ├── 05_regression_plots.ipynb         # regplot, lmplot, residplot
│   ├── 06_heatmaps.ipynb                 # heatmap, clustermap
│   ├── 07_facetgrid_pairgrid.ipynb       # Multi-plot grids, pairplot
│   └── 08_seaborn_objects.ipynb          # New Seaborn Objects API (v0.12+)
│
├── 📁 05_scipy/
│   ├── 01_statistics.ipynb               # Distributions, tests, descriptive
│   ├── 02_hypothesis_testing.ipynb       # t-test, ANOVA, chi-square, Mann-Whitney
│   ├── 03_optimization.ipynb             # minimize, curve_fit, root finding
│   ├── 04_interpolation.ipynb            # interp1d, griddata, splines
│   ├── 05_integration.ipynb              # quad, dblquad, solve_ivp
│   ├── 06_signal_processing.ipynb        # Filters, FFT, peak finding
│   ├── 07_linear_algebra.ipynb           # LU, QR, SVD, Cholesky
│   └── 08_spatial_and_cluster.ipynb      # KD-trees, Voronoi, hierarchical
│
├── 📁 06_projects/
│   ├── 01_eda_full_pipeline.ipynb        # End-to-end Exploratory Data Analysis
│   ├── 02_sales_dashboard.ipynb          # Business analytics & KPI charts
│   ├── 03_statistical_analysis.ipynb     # Real hypothesis testing case study
│   ├── 04_time_series_analysis.ipynb     # Forecasting & trend decomposition
│   ├── 05_signal_analysis.ipynb          # Audio/signal processing project
│   └── 06_geospatial_intro.ipynb         # Geographic data visualization
│
├── 📁 07_cheatsheets/
│   ├── numpy_cheatsheet.md
│   ├── pandas_cheatsheet.md
│   ├── matplotlib_cheatsheet.md
│   ├── seaborn_cheatsheet.md
│   └── scipy_cheatsheet.md
│
├── 📁 data/
│   ├── raw/                              # Raw datasets (CSV, Excel, JSON)
│   ├── processed/                        # Cleaned datasets
│   └── README.md                         # Dataset descriptions & sources
│
├── 📁 outputs/
│   ├── figures/                          # Saved plots (PNG, SVG, PDF)
│   └── reports/                          # Generated reports
│
├── 📄 requirements.txt                   # pip dependencies
├── 📄 environment.yml                    # conda environment
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 README.md
```

<br />

---

## 📓 Notebooks at a Glance

| # | Notebook | Library | Level | Topics |
|---|---|---|---|---|
| 01 | Array Basics | NumPy | 🟢 Beginner | Creation, dtype, shape, attributes |
| 02 | Broadcasting | NumPy | 🟡 Intermediate | Rules, vectorization, efficiency |
| 03 | Linear Algebra | NumPy | 🔴 Advanced | Eigenvalues, SVD, solving systems |
| 04 | Data Cleaning | Pandas | 🟢 Beginner | Nulls, types, duplicates, strings |
| 05 | GroupBy Mastery | Pandas | 🟡 Intermediate | Aggregation, transform, apply |
| 06 | Time Series | Pandas | 🔴 Advanced | Resampling, rolling, DatetimeIndex |
| 07 | Figure & Axes | Matplotlib | 🟢 Beginner | OOP interface, subplot basics |
| 08 | Layouts | Matplotlib | 🟡 Intermediate | GridSpec, mosaic, constrained layout |
| 09 | 3D & Animations | Matplotlib | 🔴 Advanced | Surface plots, FuncAnimation |
| 10 | Distribution Plots | Seaborn | 🟢 Beginner | histplot, kdeplot, ecdfplot |
| 11 | FacetGrid | Seaborn | 🟡 Intermediate | Multi-panel, PairGrid, pairplot |
| 12 | Objects API | Seaborn | 🔴 Advanced | New declarative interface |
| 13 | Hypothesis Tests | SciPy | 🟡 Intermediate | t-test, ANOVA, chi-square |
| 14 | Optimization | SciPy | 🔴 Advanced | curve_fit, minimize, LP |
| 15 | Signal Processing | SciPy | 🔴 Advanced | FFT, filtering, spectral analysis |

<br />

---

## 📋 Quick Cheatsheets

<br />

### 🔢 NumPy — Most Used

```python
import numpy as np

# Creation
np.array([1,2,3])             # from list
np.zeros((3,4))               # all zeros
np.ones((3,4))                # all ones
np.eye(3)                     # identity matrix
np.arange(0, 10, 2)          # [0, 2, 4, 6, 8]
np.linspace(0, 1, 100)       # 100 evenly spaced points
np.random.randn(3, 4)        # standard normal

# Properties
arr.shape                     # dimensions tuple
arr.ndim                      # number of dimensions
arr.dtype                     # data type
arr.size                      # total elements

# Operations
arr.reshape(2, 6)             # new shape (must be compatible)
arr.T                         # transpose
arr.flatten()                 # 1D copy
np.concatenate([a, b], axis=0)
np.stack([a, b], axis=1)
np.split(arr, 3)

# Math
np.sum(arr, axis=0)           # sum along axis
np.mean(arr)
np.std(arr)
np.dot(A, B)                  # matrix multiplication
np.linalg.inv(A)              # inverse
np.linalg.eig(A)              # eigenvalues & vectors
```

<br />

### 🐼 Pandas — Most Used

```python
import pandas as pd

# Load
df = pd.read_csv("file.csv")
df = pd.read_excel("file.xlsx")
df = pd.read_json("file.json")
df = pd.read_parquet("file.parquet")

# Explore
df.head(10)
df.info()
df.describe()
df.dtypes
df.shape
df["col"].value_counts()
df.isnull().sum()

# Select
df["col"]                      # Series
df[["col1", "col2"]]          # DataFrame
df.loc[rows, cols]             # label-based
df.iloc[0:5, 2:4]             # position-based
df.query("age > 30 and city == 'NYC'")

# Clean
df.dropna()
df.fillna(0)
df.drop_duplicates()
df["col"].astype("int")
df["col"].str.lower().str.strip()

# Transform
df.rename(columns={"old": "new"})
df["new_col"] = df["col"] * 2
df.assign(new_col=lambda x: x["a"] + x["b"])
df.apply(func, axis=1)

# GroupBy
df.groupby("col")["val"].sum()
df.groupby("col").agg({"a": "sum", "b": "mean"})

# Merge
pd.merge(df1, df2, on="key", how="left")
pd.concat([df1, df2], axis=0, ignore_index=True)
```

<br />

### 📊 Matplotlib — Most Used

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Line
axes[0].plot(x, y, color="blue", linestyle="--", linewidth=2, label="Signal")
axes[0].set_title("Title")
axes[0].set_xlabel("X Axis")
axes[0].set_ylabel("Y Axis")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scatter
scatter = axes[1].scatter(x, y, c=z, cmap="viridis", s=50, alpha=0.7)
fig.colorbar(scatter, ax=axes[1])

# Histogram
plt.hist(data, bins=30, edgecolor="black", density=True)

# Bar
plt.bar(categories, values, color="steelblue")

# Save
plt.tight_layout()
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
plt.show()
```

<br />

### 🌊 Seaborn — Most Used

```python
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# Distribution
sns.histplot(data=df, x="value", kde=True, hue="category")
sns.kdeplot(data=df, x="value", fill=True)

# Categorical
sns.boxplot(data=df, x="category", y="value", hue="group")
sns.violinplot(data=df, x="day", y="total_bill", hue="sex", split=True)
sns.barplot(data=df, x="category", y="value", errorbar="ci")

# Relational
sns.scatterplot(data=df, x="x", y="y", hue="label", size="importance")
sns.lineplot(data=df, x="time", y="value", hue="group", style="type")

# Regression
sns.regplot(data=df, x="x", y="y", scatter_kws={"alpha": 0.5})

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)

# Pairplot
sns.pairplot(df, hue="target", diag_kind="kde")
```

<br />

### 🔬 SciPy — Most Used

```python
from scipy import stats, optimize, interpolate, signal, integrate

# Statistics
stats.ttest_ind(a, b)                          # independent t-test
stats.chi2_contingency(table)                  # chi-square test
stats.f_oneway(g1, g2, g3)                    # one-way ANOVA
stats.norm.pdf(x, loc=0, scale=1)             # PDF of normal dist
stats.norm.cdf(x)                              # CDF
stats.pearsonr(x, y)                           # Pearson correlation
stats.spearmanr(x, y)                          # Spearman correlation

# Optimization
optimize.minimize(func, x0, method="BFGS")
optimize.curve_fit(model, xdata, ydata)
optimize.brentq(func, a, b)                    # root finding

# Interpolation
f = interpolate.interp1d(x, y, kind="cubic")
y_new = f(x_new)

# Integration
result, err = integrate.quad(func, a, b)
sol = integrate.solve_ivp(ode_func, [t0, tf], y0)

# Signal
b, a = signal.butter(N=4, Wn=0.2)
y_filtered = signal.filtfilt(b, a, x)
freqs, psd = signal.welch(x, fs=1000)
peaks, _ = signal.find_peaks(x, height=0.5)
```

<br />

---

## 🧪 Real-World Projects Included

<br />

### 🔍 Project 1 — Full EDA Pipeline
A complete **exploratory data analysis** workflow on a real-world dataset:
- Load & profile raw data
- Handle missing values, outliers, data types
- Statistical summaries and distribution analysis
- Correlation analysis and feature relationships
- Full visualization dashboard with Matplotlib + Seaborn

<br />

### 📈 Project 2 — Sales Analytics Dashboard
Business intelligence analysis on retail/e-commerce data:
- Revenue trends over time (time series)
- Product category performance (GroupBy + charts)
- Regional performance maps
- Monthly/quarterly KPI visualizations
- Customer segmentation charts

<br />

### 🧮 Project 3 — Statistical Analysis Case Study
Rigorous hypothesis-driven analysis:
- A/B test design and evaluation
- Multiple hypothesis testing with corrections
- Effect size calculations (Cohen's d, eta-squared)
- Confidence intervals and power analysis with SciPy
- Visual reporting of statistical results

<br />

### 📡 Project 4 — Signal Processing
Engineering/science application:
- Generate synthetic and real-world signals
- Noise injection and filtering (Butterworth, Chebyshev)
- FFT frequency analysis and spectrograms
- Peak detection and feature extraction
- Visualization of time and frequency domains

<br />

---

## 👥 Who Is This For?

```
🧑‍🎓  BEGINNERS              Learning Python data science for the first time
🧑‍💻  DEVELOPERS             Adding data skills to their engineering toolkit
📊  DATA ANALYSTS           Upgrading from Excel to Python workflows
🔬  RESEARCHERS             Applying statistical computing to real problems
🤖  ML ENGINEERS            Building stronger foundations before modeling
🎓  STUDENTS                Following structured notebooks for coursework
```

<br />

---

## 🤝 Contributing

Contributions are warmly welcome! Whether it's fixing a bug, adding a notebook, improving documentation, or adding a new real-world project:

1. Fork the repository
2. Create your branch: `git checkout -b feature/add-scipy-ode-notebook`
3. Commit your changes: `git commit -m "Add ODE solving notebook with examples"`
4. Push to your branch: `git push origin feature/add-scipy-ode-notebook`
5. Open a Pull Request

Please follow the existing notebook format and include clear markdown explanations above each code cell.

<br />

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details. Free to use, learn from, and build upon.

<br />

---

## 📌 Resources & References

| Resource | Link |
|---|---|
| 📘 NumPy Documentation | [numpy.org/doc](https://numpy.org/doc) |
| 📗 Pandas Documentation | [pandas.pydata.org/docs](https://pandas.pydata.org/docs) |
| 📙 Matplotlib Documentation | [matplotlib.org/stable/contents](https://matplotlib.org/stable/contents.html) |
| 📕 Seaborn Documentation | [seaborn.pydata.org](https://seaborn.pydata.org) |
| 📓 SciPy Documentation | [docs.scipy.org](https://docs.scipy.org) |
| 📔 Python Data Science Handbook | [jakevdp.github.io/PythonDataScienceHandbook](https://jakevdp.github.io/PythonDataScienceHandbook/) |
| 🎓 Kaggle Learn | [kaggle.com/learn](https://kaggle.com/learn) |

<br />

---

<br />

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         Built with passion by  Hilal                            ║
║         Data Science & Development Ecosystem Creator            ║
║                                                                  ║
║         🌐  lokallhost.io                                        ║
║         👤  hila-11.com                                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

<br />

**Found this useful? Drop a ⭐ — it helps more people discover this resource.**

<br />

[![Star on GitHub](https://img.shields.io/github/stars/your-username/numpy-pandas-matplotlib-seaborn-scipy?style=for-the-badge&logo=github&label=Star%20this%20repo&color=013243)](https://github.com/your-username/numpy-pandas-matplotlib-seaborn-scipy)
[![Follow on GitHub](https://img.shields.io/github/followers/your-username?style=for-the-badge&logo=github&label=Follow&color=150458)](https://github.com/your-username)

<br />

*Happy Analyzing 🚀*

</div>
