# Kolmogorov–Arnold Networks for Thermoelectric Materials Design

Code and data accompanying

> M. Fronzi, M. Ford, K. Nayal, O. Isayev, C. Stampfl,
> **"Kolmogorov–Arnold Networks in Thermoelectric Materials Design"**,
> *Scientific Reports* **16**, 14146 (2026).

The notebooks train and compare three families of models that predict the
electronic band gap and the n- and p-type Seebeck coefficients of crystalline
materials from a learned 128-dimensional composition/structure embedding:

| Notebook | What it does |
|---|---|
| `KAN_BandGap_Sn.ipynb` | Trains Kolmogorov–Arnold Networks ([pykan](https://github.com/KindXiaoming/pykan)) with k-fold cross-validation, saves best checkpoints and fold metrics, and extracts node/edge/feature importance scores that make the learned structure–property relationships inspectable. |
| `MLP_BandGap_Seebeck.ipynb` | Multilayer-perceptron baseline for the same three targets (`band_gap`, `s_n`, `s_p`) with grid search, and symbolic post-processing of the fitted KAN activation functions with `sympy`. |
| `Conventional_ML_BandGap_Seebeck.ipynb` | Conventional regressors as a reference point: random forest, kernel ridge, SVR, scikit-learn MLP, XGBoost, LightGBM, CatBoost, with randomised hyper-parameter search. |
| `plot_maps.ipynb` | Produces the 2-D response heatmaps (pairs of input features against predicted property) used in the figures. |

## Data

`train_embeddings_250epochs_embdim128.pt` (19 MB) is a PyTorch tensor holding
the 128-dimensional embedding of each material followed by three target
columns: the band gap (third-from-last column) and the n- and p-type Seebeck
coefficients (last two columns; see the `y_dict` definition at the top of
each notebook for the column assignment). The
conventional-ML notebook reads the same data from a CSV export
(`train_embeddings_250epochs_embdim128_original.csv`); regenerate it from the
`.pt` file with:

```python
import torch, pandas as pd
x = torch.load("train_embeddings_250epochs_embdim128.pt").numpy()
pd.DataFrame(x).to_csv("train_embeddings_250epochs_embdim128_original.csv", index=False)
```

## Running

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks from the repository root. They write all intermediate
files (scalers, checkpoints, per-fold metrics, predictions, figures) to a
`results/` folder that is created on first run and is git-ignored.

Order: `KAN_BandGap_Sn.ipynb` and `MLP_BandGap_Seebeck.ipynb` are independent;
`plot_maps.ipynb` reads trained models from `results/`.

## Citation

```bibtex
@article{Fronzi2026KAN,
  author  = {Fronzi, Marco and Ford, Michael and Nayal, Kamal and Isayev, Olexandr and Stampfl, Catherine},
  title   = {Kolmogorov--Arnold Networks in Thermoelectric Materials Design},
  journal = {Scientific Reports},
  volume  = {16},
  pages   = {14146},
  year    = {2026}
}
```

## Contact

Marco Fronzi, School of Physics, The University of Sydney — marco.fronzi@sydney.edu.au
