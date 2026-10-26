# Code

Everything in this folder is real code from the thesis project. Nothing was rewritten after the fact; files were only copied, renamed and cleaned as described below.

## `notebooks/`

The Google Colab notebooks I used for training and evaluation (spring 2025). Several of them started from the public Roboflow / Ultralytics training notebook templates, which is why they contain tutorial cells next to my own code.

| File | Original name in my Drive | What it does |
|---|---|---|
| `01_train_yolov5s_50_epochs.ipynb` | yolov5-mergeddataset_metnull_50epoch.ipynb | Trains YOLOv5s on the merged dataset (50 epochs, 640 px, batch 16). |
| `02_train_yolov8s_50_epochs.ipynb` | yolov8-mergeddataset_withnull.ipynb | Trains YOLOv8s on the same dataset (50 epochs, 800 px). |
| `03_train_yolov8_sqrt_weighted_sampling.ipynb` | train_yolov8_weightedclasses | Builds the `YOLOWeightedDataset` (inverse, SQRT and "smart" variants), checks the sampling, simulates class exposure and trains YOLOv8s with weighted sampling. |
| `04_evaluate_yolov5_confidence_and_heatmaps.ipynb` | yolov5_results_heatmap_ground_truth | Confidence-vs-ground-truth plots, confusion matrix, precision/recall/accuracy and heatmap experiments for YOLOv5. |
| `05_evaluate_yolov8_confidence_and_eigencam.ipynb` | Untitled0.ipynb | The same evaluation for the YOLOv8 models, plus EigenCAM heatmaps with YOLOv8_Explainer. |

Cleaning applied:
- The Roboflow API key was replaced by `YOUR_ROBOFLOW_API_KEY`.
- Image outputs (training batches, predictions on dataset images) were removed; text outputs such as training logs and metrics were kept, with very long logs trimmed.
- Colab-specific metadata was removed.

The notebooks expect the datasets to be downloaded from Roboflow and the weights to be on Google Drive, so they will not run as-is without those.

## `weighted_dataset_sqrt.py` and `weighted_dataset_smart.py`

The class-balancing sampler, copied out of notebook 03:
- `weighted_dataset_sqrt.py`: per-class weight `sqrt(N_total / N_class)`, normalised; each image gets the mean weight of its classes; images are drawn with those probabilities during training only. This is the version described in chapter 5 of the thesis.
- `weighted_dataset_smart.py`: a later variant from the same notebook that starts from the SQRT weights and adds caps and floors per class (for example `damage` at least 0.8, `Big-Damage` at least 3.0).

Usage, as in the notebook: define the class, then replace Ultralytics' dataset class before calling train:

```python
import ultralytics.data.build as build
build.YOLODataset = YOLOWeightedDataset
```

## `thesis-listings/`

The code fragments ("codefragmenten") printed in the thesis, listings 5.1 to 5.5.

## Not available

The live detection scripts used during the field test (visible in a thesis screenshot as `rtmp_detect.py` and `detect_and_capture_v8.py`) and the `mediamtx.yml` config could not be recovered, so they are not included. Chapter 6 of the thesis describes how the detection loop works.
