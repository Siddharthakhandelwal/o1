# RadioSSL

Code and Kaggle workflow for the RadioSSL ILD CT patch representation project.

## Run on Kaggle

Open `RadioSSL_Kaggle_runbook.ipynb` in Kaggle, attach the extracted ILD dataset, enable Internet and a GPU, and run notebook cells in order. The notebook discovers `ILD_DB_volumeROIs` under `/kaggle/input`, combines Talisman and volume/pilot patches, and performs a patient-grouped 75/25 split before SSL. All rows for a matching patient ID stay together; patient `-1` sentinel rows are always training-only.

Do not commit the source dataset, protected clinical workbook, generated NPZ files, checkpoints, or training logs.
