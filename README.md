# Hosted ESCI Product Reranker Pipeline

This folder contains the complete pipeline for fine-tuning one model using only a public dataset.

```text
Dataset: tasksource/esci (processed mirror of Amazon ESCI)
Source: https://github.com/amazon-science/esci-data
Model: cross-encoder/ms-marco-MiniLM-L6-v2
Training: Google Colab, Kaggle, or a Hugging Face GPU Space
Hosting: Hugging Face Inference Endpoint
```

The public mirror is already joined with product metadata. No GenieAI/private examples are used.

## Pipeline

```text
prepare_data.py
    → downloads English ESCI data
    → keeps queries from 10 gift categories by default
    → keeps complete query groups
    → creates query/product/label pairs

train.py
    → fine-tunes MiniLM
    → saves the model
    → optionally uploads it to a private Hugging Face repository

evaluate.py
    → calculates NDCG@10, MRR and HitRate@4

publish_endpoint.py
    → adds the hosted inference handler to the model repository

call_endpoint.py
    → tests the deployed HTTPS endpoint
```

## 1. Run on Hosted Compute

Open Google Colab or Kaggle with a GPU, upload this folder, and enter it:

```bash
cd reranker-pipeline
pip install -r requirements.txt
```

For Hugging Face upload, create a write token and save it as a notebook secret/environment variable:

```bash
export HF_TOKEN=hf_your_write_token
```

Never commit the token.

## 2. Prepare Public ESCI Data

```bash
python prepare_data.py \
  --train-pairs 100000 \
  --validation-pairs 10000
```

Default gift categories:

```text
cakes and desserts
flower bouquets
chocolates and candy
perfume and fragrance
jewelry
fashion and accessories
gift baskets and hampers
skincare and beauty sets
personalized gifts
home decor and candles
```

The category is detected from the query. All products belonging to that query—including irrelevant negatives—are retained for correct ranking training. Each prepared row includes `gift_category`.

To prepare the complete general ESCI dataset instead:

```bash
python prepare_data.py --scope all
```

Output:

```text
data/processed/
```

Labels are converted as follows:

```text
Exact = 1.00
Substitute = 0.70
Complement = 0.35
Irrelevant = 0.00
```

The official ESCI train/test split is retained. Sampling keeps all candidates belonging to a selected query together.

## 3. Fine-Tune and Upload

Replace `YOUR_USERNAME`:

```bash
python train.py \
  --hub-repo YOUR_USERNAME/genieai-product-reranker
```

Default training settings:

```text
Epochs: 2
Batch size: 32
Learning rate: 2e-5
Maximum input length: 384
```

If GPU memory is insufficient:

```bash
python train.py \
  --batch-size 8 \
  --hub-repo YOUR_USERNAME/genieai-product-reranker
```

The trained model is saved at:

```text
models/genieai-product-reranker/final/
```

## 4. Evaluate

Fine-tuned model:

```bash
python evaluate.py
```

Public base-model baseline:

```bash
python evaluate.py \
  --model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --output results/base-model-metrics.json
```

The fine-tuned model should improve ranking metrics over the base model.

## 5. Publish Endpoint Files

```bash
python publish_endpoint.py \
  --repo-id YOUR_USERNAME/genieai-product-reranker
```

This uploads `handler.py` and its dependency file to the model repository.

## 6. Deploy the Hosted Model

1. Open [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/quick_start).
2. Create a protected endpoint.
3. Select `YOUR_USERNAME/genieai-product-reranker`.
4. Start with CPU hardware.
5. Deploy and copy the endpoint URL.

Hugging Face hosts the model. GenieAI only calls its HTTPS endpoint.

## 7. Test the Endpoint

```bash
export HF_RERANKER_ENDPOINT_URL=https://your-endpoint.endpoints.huggingface.cloud
export HF_RERANKER_TOKEN=hf_your_read_token
python call_endpoint.py
```

Expected result: the flower product ranks above the headphones product.

## 8. GenieAI Request Format

Send the normalized English query and 30 RAG candidates:

```json
{
  "inputs": {
    "query": "birthday flowers for mother",
    "top_n": 4,
    "documents": [
      {
        "id": "product-123",
        "text": "Pink rose bouquet arranged for birthdays"
      }
    ]
  }
}
```

Response:

```json
{
  "results": [
    {
      "id": "product-123",
      "score": 5.82,
      "original_index": 7
    }
  ]
}
```

Integration order:

```text
English searchQuery
→ RAG retrieves 30 products
→ hard budget and stock filters
→ hosted reranker
→ top 4 products
→ Groq generates the chat response
```

If the endpoint fails or times out, keep the original RAG order.

## Notes

- Do not reduce candidates to four before reranking.
- MCP remains the source of truth for live price, stock and delivery.
- Raw model scores are ranking values, not guaranteed probabilities.
- ESCI is Apache 2.0 licensed; retain its attribution and citation.
