# Run Local Delivery-Time Predictions

Use this after training has created `models/delivery-time-regressor/model.joblib` and `metadata.json`.

## 1. Install dependencies

```powershell
cd E:\Projects\GenieAI-ML\delivery-prediction-pipeline\colab_hf
python -m pip install -r requirements.txt
```

## 2. Create and run a prediction

Run this from the `colab_hf` folder. Replace the example values with the delivery details you want to estimate.

```powershell
@'
import json
from pathlib import Path

import joblib
import pandas as pd

model_dir = Path("models/delivery-time-regressor")
model = joblib.load(model_dir / "model.joblib")
metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

# Every feature recorded in metadata.json must be provided. Use null for unknown values.
delivery = {
    "delivery_person_age": 30,
    "delivery_person_ratings": 4.7,
    "restaurant_latitude": 22.745049,
    "restaurant_longitude": 75.892471,
    "delivery_location_latitude": 22.765049,
    "delivery_location_longitude": 75.912471,
    "weather_conditions": "Sunny",
    "road_traffic_density": "Low",
    "vehicle_condition": 2,
    "type_of_order": "Meal",
    "type_of_vehicle": "motorcycle",
    "multiple_deliveries": 0,
    "festival": "No",
    "city": "Metropolitian",
    "distance_km": 3.1,
    "day_of_week": "Monday",
    "month": 3,
    "hour_of_day": 14
}

# Keep only the fields expected by this exact trained model, in its saved order.
row = {column: delivery.get(column) for column in metadata["feature_columns"]}
minutes = float(model.predict(pd.DataFrame([row]))[0])
result = {"predicted_delivery_time_minutes": round(minutes, 2)}

Path("prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
'@ | python
```

This prints the prediction and saves it as `prediction.json` in the current folder:

```json
{
  "predicted_delivery_time_minutes": 32.45
}
```

## Important input notes

- The model only accepts the feature names in `models/delivery-time-regressor/metadata.json`. The script above filters the example to those names automatically.
- Use the same values/labels seen during training where possible, such as `Sunny`, `Low`, `motorcycle`, and `Metropolitian` (the dataset's original spelling).
- Unknown categorical labels are accepted, but an estimate is most reliable for conditions represented in the training data.
- `distance_km`, `day_of_week`, `month`, and `hour_of_day` are created during data preparation. For new deliveries, calculate/provide them before prediction.

## Predict several deliveries

Place a JSON array of delivery objects in `requests.json`, then replace `delivery` above with:

```python
requests = json.loads(Path("requests.json").read_text(encoding="utf-8"))
rows = [{column: item.get(column) for column in metadata["feature_columns"]} for item in requests]
predictions = model.predict(pd.DataFrame(rows))
print([round(float(value), 2) for value in predictions])
```
