# RadioSSL

Code and Kaggle workflow for the RadioSSL ILD CT patch representation project.

## Run on Kaggle

1. Add the original `ILD_DB.zip` as a Kaggle Dataset input.
2. Open `RadioSSL_Kaggle_runbook.ipynb` in Kaggle and enable Internet and a P100 GPU.
3. Set `DATASET_ZIP` to the attached archive path, then run notebook cells in order.

The preprocessing creates separate training and validation NPZ files. Volume ROI and HRCT pilot cases are used for training. Talisman patients whose numeric IDs also occur in training are excluded from validation; sentinel `-1` patches are excluded unless explicitly enabled.

Do not commit the source dataset, protected clinical workbook, generated NPZ files, checkpoints, or training logs.
