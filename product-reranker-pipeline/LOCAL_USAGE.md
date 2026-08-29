# Use the GenieAI Reranker Locally

Run the model in a Python backend. Do not load it in the Next.js browser frontend.

## 1. Install

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install sentence-transformers torch fastapi uvicorn
```

## 2. Choose the Model Source

Private Hugging Face repository:

```powershell
$env:HF_TOKEN="hf_your_read_token"
$env:RERANKER_MODEL="ramitha2002/genieai-product-reranker"
```

Or use the extracted local model folder:

```powershell
$env:RERANKER_MODEL="C:\models\genieai-product-reranker\final"
```

## 3. Create `local_reranker.py`

```python
import os
from typing import Any

from sentence_transformers import CrossEncoder


MODEL_SOURCE = os.environ.get(
    "RERANKER_MODEL",
    "ramitha2002/genieai-product-reranker",
)

model = CrossEncoder(
    MODEL_SOURCE,
    token=os.environ.get("HF_TOKEN"),
)


def build_product_text(product: dict[str, Any]) -> str:
    fields = [
        ("Title", product.get("title")),
        ("Description", product.get("description")),
        ("Features", product.get("features")),
        ("Brand", product.get("brand")),
        ("Color", product.get("color")),
    ]
    return "\n".join(
        f"{name}: {value}" for name, value in fields if value
    )


def rerank(
    query: str,
    products: list[dict[str, Any]],
    top_n: int = 4,
) -> list[dict[str, Any]]:
    if not products:
        return []

    pairs = [
        (query, build_product_text(product))
        for product in products
    ]
    scores = model.predict(pairs, batch_size=16)

    scored_products = [
        {**product, "rerankerScore": float(score)}
        for product, score in zip(products, scores)
    ]
    return sorted(
        scored_products,
        key=lambda product: product["rerankerScore"],
        reverse=True,
    )[:top_n]


if __name__ == "__main__":
    sample_products = [
        {
            "id": "flowers-1",
            "title": "Pink Rose Bouquet",
            "description": "Fresh roses arranged for birthdays",
            "brand": "Bloom House",
            "color": "Pink",
        },
        {
            "id": "mouse-1",
            "title": "Wireless Gaming Mouse",
            "description": "RGB computer mouse",
            "brand": "GamePoint",
            "color": "Black",
        },
    ]

    results = rerank(
        "birthday flowers for mother",
        sample_products,
    )
    for result in results:
        print(result["id"], result["rerankerScore"])
```

Test it:

```bash
python local_reranker.py
```

The flower product should rank above the gaming mouse.

## 4. Optional Local HTTP API

Create `api.py` beside `local_reranker.py`:

```python
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from local_reranker import rerank


app = FastAPI()


class RerankRequest(BaseModel):
    query: str
    products: list[dict[str, Any]]
    top_n: int = Field(default=4, ge=1, le=30)


@app.post("/rerank")
def rerank_products(request: RerankRequest):
    return {
        "results": rerank(
            request.query,
            request.products,
            request.top_n,
        )
    }
```

Start it:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Request format:

```json
{
  "query": "birthday flowers for mother",
  "top_n": 4,
  "products": [
    {
      "id": "flowers-1",
      "title": "Pink Rose Bouquet",
      "description": "Fresh roses arranged for birthdays",
      "brand": "Bloom House",
      "color": "Pink"
    }
  ]
}
```

The local endpoint is:

```text
http://127.0.0.1:8000/rerank
```

## GenieAI Flow

```text
RAG retrieves about 30 products
→ GenieAI backend calls the local Python endpoint
→ CrossEncoder scores the products
→ return the best 4
```

Keep the original RAG order as a fallback if the local service is unavailable.
