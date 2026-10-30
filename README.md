# SAGE – Autonomous Visual AI Agent for Precision Agriculture
## Multimodal Image + Text Vision AI Inference API Backend

Production-grade, modular FastAPI inference backend serving crop disease diagnoses using **Qwen2.5-VL-3B-Instruct** combined with crop-specific Parameter-Efficient Fine-Tuned (PEFT) LoRA adapters.

Supports **multimodal inputs (Image + Text prompt)**:
1. **Crop name** -> Activates the corresponding crop LoRA adapter.
2. **Plant image** -> Visual evidence processed natively by Qwen vision processor.
3. **User prompt / instruction (Optional)** -> Custom query directing the model's diagnostic focus (e.g. disease ID, symptom explanation, severity, management advice).

Designed for cloud GPU instances such as **Kaggle Notebooks (Tesla T4 / P100)** and **Google Colab**, exposed externally through an encrypted high-speed tunnel.

---

## Table of Contents
1. [Multimodal Architecture Overview](#multimodal-architecture-overview)
2. [Hugging Face Checkpoint Mapping](#hugging-face-checkpoint-mapping)
3. [Crop Registry & Default Prompts](#crop-registry--default-prompts)
4. [Flexible User Prompts & Limitations](#flexible-user-prompts--limitations)
5. [Primary API Endpoints & Request Parameters](#primary-api-endpoints--request-parameters)
6. [Structured JSON Response Format](#structured-json-response-format)
7. [Working cURL Examples](#working-curl-examples)
8. [n8n Workflow Integration](#n8n-workflow-integration)
9. [External Application Integration](#external-application-integration)
10. [Adapter Switching & Concurrency Synchronization](#adapter-switching--concurrency-synchronization)
11. [Kaggle & Colab Deployment Guides](#kaggle--colab-deployment-guides)
12. [Testing & Mock Mode](#testing--mock-mode)

---

## Multimodal Architecture Overview

SAGE leverages the native multimodal capabilities of **Qwen2.5-VL-3B-Instruct**:
- **Base Model**: Loaded into GPU memory **only once** upon server startup.
- **Crop Adapters**: PEFT LoRA adapters dynamically attached from `IDK-Chaitanya/SAGE5Class-cp`.
- **Multimodal Pipeline**:
  ```mermaid
  flowchart LR
    A["Request: Crop + Image + Prompt"] --> B["Crop Registry Validation"]
    B --> C["Activate Crop LoRA Adapter"]
    A --> D["Image Normalization (RGB, EXIF)"]
    C --> E["Qwen2.5-VL Multimodal Input: Messages [Image + Prompt]"]
    D --> E
    E --> F["AutoProcessor & Model Generate"]
    F --> G["Token Slice & Output Parser"]
    G --> H["Structured JSON Response"]
  ```

---

## Hugging Face Checkpoint Mapping

Hugging Face Checkpoints Repository: **`IDK-Chaitanya/SAGE5Class-cp`**

| Crop ID | Display Name | Checkpoint Subpath | Active Adapter |
| :--- | :--- | :--- | :--- |
| `corn` | Corn (Maize) | `corn_5ep/checkpoints/checkpoint-step-1000` | `corn_lora` |
| `rice` | Rice | `rice/checkpoints/checkpoint-step-600` | `rice_lora` |
| `soybean` | Soybean | `soybean_5ep/checkpoints/checkpoint-step-1000` | `soybean_lora` |
| `tomato` | Tomato | `tomato/checkpoints/checkpoint-step-600` | `tomato_lora` |
| `wheat` | Wheat | `wheat/checkpoints/checkpoint-step-500` | `wheat_lora` |

---

## Crop Registry & Default Prompts

Configured centrally in [`configs/crops.yaml`](file:///c:/Projects/Sage-backend/configs/crops.yaml). If the `prompt` parameter is omitted by the client, SAGE automatically injects the tailored default prompt for that crop:

| Crop | Configured Default Prompt |
| :--- | :--- |
| **Corn** | `Identify the disease visible in this corn leaf image. Use the disease classes supported by the corn model. Provide the predicted disease and a concise explanation of the visible symptoms.` |
| **Rice** | `Identify the disease visible in this rice leaf image. Use the disease classes supported by the rice model. Provide the predicted disease and a concise explanation of the visible symptoms.` |
| **Soybean** | `Identify the disease visible in this soybean leaf image. Use the disease classes supported by the soybean model. Provide the predicted disease and a concise explanation of the visible symptoms.` |
| **Tomato** | `Identify the disease visible in this tomato leaf image. Use the disease classes supported by the tomato model. Provide the predicted disease and a concise explanation of the visible symptoms.` |
| **Wheat** | `Identify the disease visible in this wheat leaf image. Use the disease classes supported by the wheat model. Provide the predicted disease and a concise explanation of the visible symptoms.` |

---

## Flexible User Prompts & Limitations

Clients can pass arbitrary user prompts to guide the vision-language reasoning:
- **Disease Identification**: `"Identify the disease affecting this tomato leaf."`
- **Symptom Explanation**: `"Describe the visible lesions and discoloration in this image."`
- **Severity Assessment**: `"Assess the visible severity of the leaf infection."`
- **Agricultural Advice**: `"Suggest suitable disease-management practices for the identified condition."`

> [!WARNING]
> **Training Data & Domain Limitations:**
> While Qwen2.5-VL-3B supports general reasoning, the PEFT adapters were fine-tuned specifically on grounded disease classification and symptom attribution from the SAGE dataset. Custom prompts asking for extensive chemical dosage recommendations or off-domain assessments should be validated experimentally. Prediction outputs are always returned as generated by the model without fabricated probabilities.

---

## Primary API Endpoints & Request Parameters

### 1. Primary Multimodal Endpoint: `POST /api/v1/predict`
Accepts `multipart/form-data`:

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `crop` | Form String | **Yes** | Target crop (`corn`, `rice`, `soybean`, `tomato`, `wheat`) |
| `image` | UploadFile | **Yes** | Leaf/plant image file (JPEG, PNG, WebP) |
| `prompt` | Form String | **No** | Custom instruction. If omitted, uses crop default prompt. |
| `max_new_tokens`| Form Int | No | Max tokens to generate (default: 256) |
| `temperature` | Form Float | No | Sampling temperature (default: 0.0 deterministic) |

### 2. Path-based Convenience Endpoint: `POST /api/v1/predict/{crop}`
Accepts `image` file and optional `prompt` as form field.

### 3. Sequential Batch Endpoint: `POST /api/v1/predict/batch`
Accepts `crop`, multiple `images`, and optional common `prompt`.

### 4. System Endpoints:
- `GET /health`: Operational status and base model state.
- `GET /api/v1/crops`: List of supported crops, default prompts, and loading status.
- `GET /api/v1/model/status`: Active base model, active adapter, and GPU VRAM consumption.

---

## Structured JSON Response Format

```json
{
  "success": true,
  "crop": "tomato",
  "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "active_adapter": "tomato",
  "checkpoint": "tomato/checkpoints/checkpoint-step-600",
  "input": {
    "prompt": "Identify the disease visible in this tomato leaf image. Use the disease classes supported by the tomato model. Provide the predicted disease and a concise explanation of the visible symptoms."
  },
  "prediction": {
    "response": "Crop: Tomato\nDiagnosis: Early Blight\nDisease type: Fungal\nPlant organ: Foliar\nVisual symptoms: Concentric dark brown rings with target-like appearance.\nPathogen: Alternaria solani",
    "class_name": "Early Blight",
    "crop": "Tomato",
    "disease_type": "Fungal",
    "plant_organ": "Foliar",
    "visual_symptoms": "Concentric dark brown rings with target-like appearance.",
    "pathogen": "Alternaria solani",
    "matched_taxonomy_class": "Early_Blight",
    "is_canonical_match": true
  },
  "raw_output": "Crop: Tomato\nDiagnosis: Early Blight\nDisease type: Fungal\nPlant organ: Foliar\nVisual symptoms: Concentric dark brown rings with target-like appearance.\nPathogen: Alternaria solani",
  "inference_time_seconds": 2.14,
  "image_metadata": {
    "width": 1024,
    "height": 768,
    "original_format": "JPEG",
    "channels": 3
  }
}
```

---

## Working cURL Examples

Replace `https://YOUR-TUNNEL-URL` with your active Cloudflare Tunnel URL.

### 1. Multimodal Prediction with Default Prompt (Omitted prompt)
```bash
curl -X POST "https://YOUR-TUNNEL-URL/api/v1/predict" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "crop=tomato" \
  -F "image=@tomato_leaf.jpg"
```

### 2. Multimodal Prediction with Custom Prompt
```bash
curl -X POST "https://YOUR-TUNNEL-URL/api/v1/predict" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "crop=tomato" \
  -F "image=@tomato_leaf.jpg" \
  -F "prompt=Identify the disease affecting this tomato leaf and describe the visible concentric ring lesions."
```

### 3. Diagnose via Path Convenience
```bash
curl -X POST "https://YOUR-TUNNEL-URL/api/v1/predict/corn" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@corn_blight.jpg" \
  -F "prompt=Identify any fungal or bacterial disease present."
```

---

## n8n Workflow Integration

To integrate SAGE into an automated agricultural workflow in **n8n**:

1. **HTTP Request Node**:
   - **Method**: `POST`
   - **URL**: `https://YOUR-TUNNEL-URL/api/v1/predict`
   - **Authentication**: Header Auth (`Authorization: Bearer YOUR_API_KEY`)
   - **Body Content Type**: `Form-Data (multipart/form-data)`
   - **Form Fields**:
     - `crop` (Text): `{{ $json.crop_name }}`
     - `prompt` (Text, optional): `{{ $json.user_question || '' }}`
     - `image` (Binary Data): `data`
2. **Switch Node**:
   - Condition: `{{ $json.prediction.class_name }} != "Healthy"` -> Route automated advisory alert to farmers via Telegram, SMS, or WhatsApp.

---

## External Application Integration

### Python Example
```python
import requests

API_URL = "https://YOUR-TUNNEL-URL/api/v1/predict"
HEADERS = {"Authorization": "Bearer YOUR_API_KEY"}

files = {"image": open("leaf.jpg", "rb")}
data = {
    "crop": "wheat",
    "prompt": "Identify any symptoms of stripe rust on this wheat leaf."
}

response = requests.post(API_URL, headers=HEADERS, data=data, files=files)
result = response.json()
print("Model Response:\n", result["prediction"]["response"])
```

### TypeScript / JavaScript Example
```typescript
const formData = new FormData();
formData.append("crop", "soybean");
formData.append("image", fileInput.files[0]);
formData.append("prompt", "Assess the visible symptoms and diagnose any condition.");

const response = await fetch("https://YOUR-TUNNEL-URL/api/v1/predict", {
  method: "POST",
  headers: { "Authorization": "Bearer YOUR_API_KEY" },
  body: formData,
});
const result = await response.json();
console.log(result.prediction.response);
```

---

## Adapter Switching & Concurrency Synchronization

- **Persistent Base Model**: The 3B vision-language model stays loaded in VRAM.
- **Dynamic Adapter Switching**: When switching between crops (e.g. `corn` to `tomato`), the previous adapter is detached and the requested PEFT LoRA adapter is attached.
- **Asynchronous & Thread-Safe Lock**: Protected with `asyncio.Lock` and thread-level mutexes so that concurrent API requests for different crops cannot interleave adapter state or produce corrupted inferences.

---

## Kaggle & Colab Deployment Guides

Deployment notebooks are pre-built in [`notebooks/`](file:///c:/Projects/Sage-backend/notebooks/):
- **Kaggle**: [`notebooks/kaggle_deployment.ipynb`](file:///c:/Projects/Sage-backend/notebooks/kaggle_deployment.ipynb)
- **Google Colab**: [`notebooks/colab_deployment.ipynb`](file:///c:/Projects/Sage-backend/notebooks/colab_deployment.ipynb)

Both notebooks configure GPU acceleration, authenticate with Hugging Face (`HF_TOKEN`), start the FastAPI server, and expose the multimodal endpoints via Cloudflare Tunnel.

---

## Testing & Mock Mode

Test the entire multimodal API locally without a GPU or downloading weights:
```bash
# In .env set:
MOCK_INFERENCE=true

# Run unit tests:
python -m unittest discover tests
```
The test suite covers:
- Missing crop rejection
- Invalid crop handling
- Missing image validation
- Default prompt fallback
- Custom prompt acceptance
- Correct crop-to-adapter mapping
- Image preprocessing
- Multimodal message structure
- Mock inference output
- API schema serialization
