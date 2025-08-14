## Multivariate Quantum Reservoir Computing for Flood Risk (Tavis–Cummings Inspired)

This repository contains a Jupyter notebook (`qrc.ipynb`) implementing a multivariate Quantum Reservoir Computing (QRC) model for flood-risk prediction using rainfall time series. The quantum reservoir is inspired by the Tavis–Cummings model: a single bosonic cavity mode coupled to multiple two-level atoms, driven by multiple classical input channels.

Got inspiration from paper: [add link and citation here]

### What this project does
- Builds a quantum reservoir (cavity + atoms) with QuTiP
- Drives it with multiple synchronized channels from rainfall data:
  - Current Jun–Sep rainfall
  - Current Annual rainfall
  - Jun–Sep rainfall lag-1
  - Jun–Sep rainfall lag-2
- Records quantum observables (cavity quadratures, atomic Pauli-X/Y)
 - Trains a simple linear readout on these observables (with polynomial features) to predict flood risk (binary), using Ridge regression by default

---

## 1) Dataset and Preprocessing

- Input CSV: `cleaned_rainfall_flood_prediction.csv`
  - Columns: `SUBDIVISION`, `YEAR`, `Jun-Sep`, `ANNUAL`
  - Each row = one subdivision (state) in a specific year

Steps performed in `qrc.ipynb`:
- Filter to a single state: `df_s = df[df["SUBDIVISION"] == state]` and sort by `YEAR`
- Build standardized channels (z-score) of equal length T using lags L=2:
  - `u_jun_0 = z(series_jun[t])` for t ≥ L
  - `u_ann_0 = z(series_ann[t])` for t ≥ L
  - `u_jun_l1 = z(series_jun[t-1])` for t ≥ L
  - `u_jun_l2 = z(series_jun[t-2])` for t ≥ L
- Labels for classification:
  - Compute a per-state flood threshold: 80th percentile of `Jun-Sep`
  - `flood_labels[t] = 1 if series_jun[t] > threshold else 0`
  - Labels are aligned to start at index L (same as channels)

Tip: The notebook prints a table of available states and counts so you can pick one with enough data (≥100 recommended).

---

## 2) Architecture Flow Diagrams

### Overall System Architecture
```
Raw CSV Data (3996 records, 36 states, 1901-2015)
       ↓
[State Filter] → Single state (e.g., TAMIL NADU: 115 records)
       ↓
[Lag Processing] → Remove L=2 samples → 113 aligned samples
       ↓
[Build 4 Channels] → Standardized time series:
       ↓              • u_jun_0: current Jun-Sep z-score
       ↓              • u_ann_0: current Annual z-score  
       ↓              • u_jun_l1: Jun-Sep lag-1 z-score
       ↓              • u_jun_l2: Jun-Sep lag-2 z-score
       ↓
[Data Split] → Warmup: 28 | Train: 51 | Test: 34
       ↓
┌─────────────────────────────────────────────────────────┐
│                 QUANTUM RESERVOIR                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Cavity    │    │   Atom 1    │    │   Atom 2    │ │
│  │ (5 photons) │◄──►│ (2 levels)  │    │ (2 levels)  │ │
│  │             │    │             │    │             │ │
│  │             │◄───┴─────────────┘    │             │ │
│  │             │◄───────────────────────┴─────────────┘ │
│  └─────────────┘                       │   Atom 3    │ │
│       ▲                                │ (2 levels)  │ │
│       │                                │             │ │
│       └────────────────────────────────┴─────────────┘ │
│                                                         │
│ H0 = cavity + atoms + Jaynes-Cummings coupling         │
│ H1 = 4 drive operators × 4 input channels              │
└─────────────────────────────────────────────────────────┘
       ↓
[Quantum Evolution] qt.mesolve() → 113 time steps
       ↓
[Readouts] → 8 observables per time step:
       ↓     • Cavity I, Q quadratures  
       ↓     • Atom σx, σy (3 atoms × 2 = 6)
       ↓
[Feature Engineering] → Polynomial degree-2 expansion:
       ↓                 8 → 44 features (linear + quadratic + interactions)
       ↓
[Ridge Regression] → α=1.0 regularization
       ↓              51 train samples → flood probability
       ↓
[Threshold] → 0.5 decision boundary → flood risk (0/1)
```

### Quantum Reservoir Details (Tavis–Cummings Inspired)

```
Hilbert Space: H = H_cavity ⊗ H_atom1 ⊗ H_atom2 ⊗ H_atom3
Total Dimension: 5 × 2 × 2 × 2 = 40

┌─────────────────────────────────────────────────────────┐
│                    DRIVE OPERATORS                     │
├─────────────────────────────────────────────────────────┤
│ H1_ops[0]: ε·i(c - c†)     ←── u_jun_0 (current Jun-Sep) │
│ H1_ops[1]: ε·(c + c†)      ←── u_ann_0 (current Annual)  │
│ H1_ops[2]: ε·σx(atom1)     ←── u_jun_l1 (Jun-Sep lag-1)  │
│ H1_ops[3]: ε·σx(atom2)     ←── u_jun_l2 (Jun-Sep lag-2)  │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│                 TIME EVOLUTION                          │
│ H(t) = H0 + Σᵢ H1_ops[i] × channels[i](t)              │
│                                                         │
│ H0 = wc·c†c + Σᵢ(wa_i·σz_i/2 + g·(c†σ⁻_i + c·σ⁺_i))   │
│                                                         │
│ Dissipation: cavity + atomic decays                    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│                   READOUTS                             │
│ e_ops = [c+c†, i(c†-c), σx₁, σy₁, σx₂, σy₂, σx₃, σy₃] │
│                                                         │
│ Output: 113×8 matrix of expectation values             │
└─────────────────────────────────────────────────────────┘
```

### Data Flow & Dimensions
```
Input Rainfall CSV
├── 36 states × ~115 years = 3996 total records
└── Filter to 1 state → 115 records

Lag Processing (L=2)
├── Original: [year₁, year₂, ..., year₁₁₅]
├── Current:  [year₃, year₄, ..., year₁₁₅] → 113 samples
├── Lag-1:    [year₂, year₃, ..., year₁₁₄] → 113 samples  
└── Lag-2:    [year₁, year₂, ..., year₁₁₃] → 113 samples

Split Strategy (Conservative for Small Data)
├── Total usable: 113 samples
├── Warmup: 28 samples (discard quantum transients)
├── Remaining: 85 samples
├── Train: 51 samples (60% of remaining)
└── Test: 34 samples (40% of remaining)

Feature Engineering
├── Quantum readouts: 51×8 → 34×8 (train → test)
├── Polynomial expansion: 8 → 44 features
├── Ridge regularization: α=1.0
└── Output: continuous → binary (threshold=0.5)
```

---

## 3) Time-Dependent Hamiltonian Assembly

QuTiP expects a list where each time-dependent term is `[operator, coeff_array]` with `len(coeff_array) = T`:

```
H = [H0,
     [H1_ops[0], channels[0]],
     [H1_ops[1], channels[1]],
     [H1_ops[2], channels[2]],
     [H1_ops[3], channels[3]]]
```

- `times = np.arange(T)` is the time axis used by the solver
- Initial state: cavity vacuum and all atoms in ground state
- Observables (readout operators): cavity quadratures and atomic σx/σy for each atom

Time evolution is computed with QuTiP’s master-equation solver:

```
result = qt.mesolve(H, psi0, times, c_ops, e_ops)
```

- `result.expect` contains the expectation values of observables at each time → a T×D array after stacking

---

## 4) Readouts and Classical Readout Training

1. Form the readout matrix:
   - `readouts = np.array(result.expect).T` → shape `(T, D)` where D is the number of observables

2. Split into warmup/train/test along time:
   - `warmup_len` removes initial transients
   - Split the remaining into `train_len` and `test_len`
   - Use the same indices to slice `flood_labels`

3. Feature engineering:
   - PolynomialFeatures with degree=2 (no bias): adds squares and pairwise interactions
   - No manual bias column is added; Ridge handles the intercept internally

4. Regression (current default): Ridge regression (L2-regularized linear model)
   - Train: `model = Ridge(alpha=1.0, fit_intercept=True); model.fit(X_poly_train, Y_train)`
   - Predict: `Y_pred_raw = model.predict(X_poly_test)`; threshold at 0.5 for class label

5. Metrics:
   - MSE on continuous predictions; accuracy and classification report on binary predictions

Why polynomial features? The quantum readouts are already nonlinear in the inputs; the polynomial expansion adds extra classical nonlinearity in the readout stage, often improving separability with a simple linear model. With small datasets, prefer Ridge (regularized) over unregularized fits.

---

## 5) Concept Notes (Plain Language)

- Ridge regression (current default)
  - Adds L2 penalty on weights: minimize `||XW − Y||^2 + λ||W||^2`
  - Stabilizes learning with many/collinear features; improves generalization on small datasets with polynomial features
  - Typical usage in this project:
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly.fit_transform(X_raw_train)
    X_poly_test  = poly.transform(X_raw_test)

    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_poly_train, Y_train)
    Y_pred_raw = model.predict(X_poly_test)
    Y_pred = (Y_pred_raw > 0.5).astype(int)
    ```

- Pseudoinverse regression (optional alternative)
  - Solves least-squares via the Moore–Penrose pseudoinverse; returns minimum‑norm solution
  - No regularization term, so it can overfit with small datasets and many features
  - Not used by default in this project

- Logistic regression (optional alternative)
  - Directly models binary classes; outputs probabilities in [0, 1]
  - To try it:
    ```python
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    Y_pred_raw = clf.predict_proba(X_test)[:, 1]
    ```

- Data leakage (why earlier “too-good” accuracy happens)
  - When information from test data influences training (e.g., mixing time order, normalizing with full-series stats), accuracy looks great but won’t generalize
  - Our time-ordered warmup → train → test split avoids this, so accuracy is lower but more honest

---

## 6) Visualizations in the Notebook

- 2×2 figure:
  - Top-left: Jun–Sep rainfall (test window) with flood threshold
  - Top-right: Annual rainfall (test window)
  - Bottom-left: Actual vs predicted flood risk (0 or 1) over test years
  - Bottom-right: Standardized input channels (u_jun_0, u_ann_0, u_jun_l1, u_jun_l2)

- Scatter plot:
  - x = actual (0/1), y = predicted probability, red dashed line at 0.5 decision boundary

---

## 7) How to Run

1) Install dependencies (example with conda):
```
conda create -n qrc python=3.10 -y
conda activate qrc
pip install qutip numpy pandas scikit-learn matplotlib jupyter
```

2) Start Jupyter and open `qrc.ipynb`:
```
jupyter notebook
```

3) Configure at the top of the notebook:
- Choose `state` (e.g., `"UTTAR PRADESH"`, `"MAHARASHTRA"`, `"WEST BENGAL"`, `"BIHAR"`)
- Optionally adjust `L` (lags), `warmup_len`, and train/test split

4) Run all cells top to bottom

Alternative: execute headless and save an executed copy
```
jupyter nbconvert --to notebook --execute qrc.ipynb --output qrc_executed.ipynb
```

---

## 8) Hyperparameters to Tune

- Reservoir
  - `N_photons`: cavity truncation (e.g., 5–8)
  - `num_atoms`: number of atoms (e.g., 3–5)
  - `wc`, `wa_list`: frequencies
  - `g`: coupling strength
  - `kappa`: dissipation scale
  - `epsilon`: drive amplitude

- Data/Features
  - `L` (lag depth)
  - `warmup_len`, train/test ratio
  - Polynomial degree: 1 (linear), 2 (current), 3 (requires more data)
  - Readout model: Ridge (default), pseudoinverse, LogisticRegression

---

## 9) Troubleshooting

- Error: “No test data available”
  - The state may have too few points. Reduce `warmup_len`, reduce train ratio, or choose a state with more data

- Error: “Feature dimension mismatch”
  - Ensure the same `PolynomialFeatures` fitted on train is used to transform test
  - Do not change `e_ops` between train and test slices

- Error: “coeff length mismatch in mesolve”
  - Ensure that every `coeff_array` has length `T = len(times)` and that all channels share the same length

---

## 10) Reproducibility Notes

- `np.random.seed(42)` is set in the notebook
- The dataset is deterministic; only the train/test slicing depends on `total_len`, `warmup_len`, and ratios
- QuTiP may print FutureWarnings about options; they do not affect results

---

## 11) References

Got inspiration from paper: [add link and citation here]

If you use this work, please also cite the relevant Tavis–Cummings / Jaynes–Cummings and Quantum Reservoir Computing literature.


