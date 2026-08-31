# Colab → Hugging Face Model Workflow

Run these commands from this folder in Google Colab. The first command downloads the public Kaggle Food Delivery Dataset; no manual CSV upload is needed.

```bash
pip install -r requirements.txt

python download_dataset.py
python export_raw_sample.py
python prepare_data.py --input data/raw/train.csv --target "Time_taken(min)"
python train.py
python evaluate.py

export HF_TOKEN=hf_your_write_token
python upload_to_hf.py --repo-id YOUR_USERNAME/delivery-time-regressor
```

Use `--private` on the upload command if required. Keep `HF_TOKEN` in a Colab secret or environment variable; never place it in a notebook cell that will be shared.

## Hardcoded model repository

To upload as part of training, edit the `HF_MODEL_REPO_ID` value at the top of `train.py` once, then run:

```bash
export HF_TOKEN=hf_your_write_token
python train.py --upload
```

The separate `upload_to_hf.py` option remains useful because it uploads the model card containing the final evaluation metrics.

To use a different local CSV instead of the Kaggle dataset:

```bash
python prepare_data.py --input /content/my_delivery_data.csv --target "Time_taken(min)"
```

## Outputs

```text
data/processed/train.csv
data/processed/test.csv
data/processed/schema.json
data/raw/sample-10.json
models/delivery-time-regressor/model.joblib
models/delivery-time-regressor/metadata.json
models/delivery-time-regressor/metrics.json
results/metrics.json
```

After upload, configure the existing Hugging Face Space with:

```text
MODEL_REPO_ID=YOUR_USERNAME/delivery-time-regressor
```
