# SAGE – Autonomous Visual AI Agent for Precision Agriculture
## Multi-Crop Vision AI Inference API Backend

Production-grade, modular FastAPI inference backend serving crop disease diagnoses using **Qwen2.5-VL-3B-Instruct** combined with crop-specific Parameter-Efficient Fine-Tuned (PEFT) LoRA adapters.

Designed for cloud GPU instances such as **Kaggle Notebooks (Tesla T4 / P100)** and **Google Colab**, exposed externally through an encrypted high-speed tunnel.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Model & Adapter Architecture](#model--adapter-architecture)
3. [Hugging Face Repository Structure](#hugging-face-repository-structure)
4. [Central Crop Registry & Taxonomy](#central-crop-registry--taxonomy)
5. [Repository Structure](#repository-structure)
6. [Single vs. Multi-Adapter Modes](#single-vs-multi-adapter-modes)
7. [Local Testing without Model Downloads (Mock Mode)](#local-testing-without-model-downloads-mock-mode)
8. [Kaggle Deployment Guide (Primary Target)](#kaggle-deployment-guide-primary-target)
9. [Google Colab Deployment Guide](#google-colab-deployment-guide)
10. [REST API Documentation & Endpoints](#rest-api-documentation--endpoints)
11. [Working cURL Examples](#working-curl-examples)
12. [n8n Workflow Integration](#n8n-workflow-integration)
13. [External App Integration (Python & JavaScript)](#external-app-integration-python--javascript)
14. [GPU Memory & Performance Optimization](#gpu-memory--performance-optimization)
15. [Troubleshooting & Common Errors](#troubleshooting--common-errors)

---

## Project Overview

SAGE (**Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis**) delivers grounded visual diagnostics for precision agriculture. Rather than using generic classification models that only predict an index label, SAGE leverages a fine-tuned Vision-Language Model (**Qwen2.5-VL-3B-Instruct**) to inspect leaf and plant imagery and generate grounded, multi-field diagnoses:
- **Crop**: Target host plant
- **Diagnosis**: Specific disease or condition name
- **Disease Type**: Fungal, Bacterial, Viral, Deficiency, or Pest
- **Plant Organ**: Foliar, Vascular, Fruit, Stem, or Root
- **Visual Symptoms**: Grounded visual description of symptoms
- **Pathogen**: Causative biological organism

---

## Model & Adapter Architecture

| Component | Specification |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-VL-3B-Instruct` |
| **Adapter Format** | PEFT LoRA (`adapter_model.safetensors`, `adapter_config.json`) |
| **LoRA Rank ($r$)** | 16 |
| **LoRA Alpha ($\alpha$)** | 32 |
| **LoRA Dropout** | 0.05 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Inference Precision** | Automatic BF16 / FP16 detection (`torch.bfloat16` if supported, else `torch.float16`) |
| **Base Model Loading** | **Loaded once** into VRAM at startup via FastAPI lifespan |

---

## Hugging Face Repository Structure

Hugging Face Checkpoints Repository: **`IDK-Chaitanya/SAGE5Class-cp`**

The repository contains five dedicated checkpoint subdirectories:
```
IDK-Chaitanya/SAGE5Class-cp/
├── corn_5ep/checkpoints/checkpoint-step-1000/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors (~148 MB)
│   ├── processor_config.json
│   ├── tokenizer.json
│   └── chat_template.jinja
├── rice/checkpoints/checkpoint-step-600/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── soybean_5ep/checkpoints/checkpoint-step-1000/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── tomato/checkpoints/checkpoint-step-600/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── wheat/checkpoints/checkpoint-step-500/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

---

## Central Crop Registry & Taxonomy

Configured centrally in [`configs/crops.yaml`](file:///c:/Projects/Sage-backend/configs/crops.yaml):

| Crop ID | Display Name | Checkpoint Subpath | Known Disease Classes |
| :--- | :--- | :--- | :--- |
| `corn` | Corn (Maize) | `corn_5ep/checkpoints/checkpoint-step-1000` | Bacterial Rot And Blight, Common Rust, Ear Rots, Gray Leaf Spot, Leaf Blight, Maize Lethal Necrosis, Maize Streak Virus, Northern Leaf Blight, Healthy |
| `rice` | Rice | `rice/checkpoints/checkpoint-step-600` | Bacterial Leaf Blight, Blast, Brown Spot, Tungro, Healthy |
| `soybean` | Soybean | `soybean_5ep/checkpoints/checkpoint-step-1000` | Bacterial Blight, Bacterial Pustule, Brown Spot, Frogeye Leaf Spot, Fusarium Disease, Herbicide Injury, Iron Deficiency Chlorosis, Potassium Deficiency, Healthy |
| `tomato` | Tomato | `tomato/checkpoints/checkpoint-step-600` | Bacterial Leaf Spot, Early Blight, Late Blight, Leaf Mold, Leaf Mosaic Virus, Septoria Leaf Blotch, Target Spot, Tomato Leaf Curl Virus, Healthy |
| `wheat` | Wheat | `wheat/checkpoints/checkpoint-step-500` | Root Rot, Rust, Septoria Leaf Blotch, Smut, Healthy |

> **Training Metadata Note:** Grounded classes are derived from the companion dataset `IDK-Chaitanya/SAGE5class`. When the model generates diagnoses, the `ResponseParser` maps outputs to the canonical taxonomy while preserving the exact generated text. No artificial confidence logits or fabricated probabilities are generated.

---

## Repository Structure

```
Sage-backend/
├── configs/
│   └── crops.yaml               # Single source of truth for crop checkpoints & taxonomy
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory, CORS, lifespan & error handlers
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py            # GET /health
│   │   ├── routes.py            # Prediction & status endpoints
│   │   └── schemas.py           # Pydantic request & response contracts
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic BaseSettings & env resolution
│   │   ├── logging.py           # Structured logging with token redaction
│   │   └── security.py          # API key auth & image validation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── crop_registry.py     # Crop catalog manager
│   │   ├── adapter_loader.py    # HF Hub checkpoint downloader & inspector
│   │   └── model_manager.py     # Model lifecycle & PEFT adapter manager
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── image_processor.py   # RGB normalization & EXIF orientation
│   │   ├── response_parser.py   # SAGE structured output parser
│   │   └── predictor.py         # End-to-end inference engine
│   └── services/
│       ├── __init__.py
│       └── prediction_service.py # Decoupled prediction & batch coordinator
├── notebooks/
│   ├── kaggle_deployment.ipynb  # Primary 12-cell Kaggle deployment notebook
│   └── colab_deployment.ipynb   # Google Colab deployment notebook
├── scripts/
│   ├── start_server.py          # Programmatic Uvicorn launcher
│   ├── verify_checkpoints.py    # HF Hub checkpoint validation utility
│   └── download_models.py       # Offline model pre-download script
├── tests/
│   ├── __init__.py
│   ├── test_crop_registry.py    # Crop registry unit tests
│   ├── test_schemas.py          # Schema & response parser tests
│   ├── test_health.py           # Health endpoint tests
│   └── test_mock_inference.py   # Complete pipeline mock test
├── .env.example                 # Environment variables template
├── .gitignore                   # Git exclusions (weights, tokens, caches)
├── requirements.txt             # Pinned production dependencies
├── Dockerfile                   # GPU containerization configuration
└── README.md                    # Comprehensive documentation
```

---

## Single vs. Multi-Adapter Modes

Configured via the environment variable `ADAPTER_MODE`:

### 1. `ADAPTER_MODE=single` (Default & Recommended)
- Loads the base model `Qwen2.5-VL-3B-Instruct` once into GPU memory.
- Attaches **only one active crop adapter** at a time (~148 MB VRAM).
- When another crop is requested:
  - Safely detaches the previous adapter.
  - Releases unused GPU cache via `torch.cuda.empty_cache()` and `gc.collect()`.
  - Attaches the newly requested crop adapter.
- **Benefits**: Minimizes VRAM (~7.5 GB total VRAM), runs reliably on 15 GB / 16 GB GPUs (Tesla T4) without memory fragmentation.
- Thread-safe adapter switching lock prevents concurrent conflicts.

### 2. `ADAPTER_MODE=multi`
- Retains all loaded crop adapters simultaneously in VRAM using PEFT multi-adapter routing.
- Fast switching via `model.set_adapter(crop_name)`.
- Recommended on GPUs with $\ge 24\text{ GB}$ VRAM (e.g. A100, RTX 4090).

---

## Local Testing without Model Downloads (Mock Mode)

You can run and test the complete API pipeline locally **without a GPU or 3B model weights**:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Enable mock mode in `.env`:
   ```ini
   MOCK_INFERENCE=true
   ```
3. Run the test suite:
   ```bash
   python -m unittest discover tests
   ```

---

## Kaggle Deployment Guide (Primary Target)

Kaggle Notebooks provide free access to NVIDIA Tesla T4 or P100 GPUs (15–16 GB VRAM).

### Quick Steps:
1. Open a new Kaggle Notebook: **Kaggle -> Create -> New Notebook**.
2. Change Accelerator: **Settings (Right panel) -> Accelerator -> GPU T4 x2 or GPU P100**.
3. Turn on Internet: **Settings -> Internet -> On**.
4. Configure Kaggle Secrets:
   - Click **Add-ons -> Secrets**.
   - Label: `HF_TOKEN`, Value: Your Hugging Face user access token.
5. Clone and launch from the provided notebook:
   - Upload or paste cells from [`notebooks/kaggle_deployment.ipynb`](file:///c:/Projects/Sage-backend/notebooks/kaggle_deployment.ipynb).

### Notebook Pipeline Execution:
- **CELL 1**: Installs requirements and downloads `cloudflared`.
- **CELL 2**: Clones repository into `/kaggle/working/Sage-backend`.
- **CELL 3**: Loads `HF_TOKEN` from Kaggle Secrets.
- **CELL 4**: Checks GPU availability and detects BF16/FP16 precision.
- **CELL 5**: Runs `python scripts/verify_checkpoints.py` to validate all 5 crop adapters on HF Hub.
- **CELL 6**: Inspects SafeTensors files and PEFT configurations.
- **CELL 7**: Loads `Qwen/Qwen2.5-VL-3B-Instruct` into GPU memory.
- **CELL 8**: Attaches initial adapter (`tomato_lora`).
- **CELL 9**: Runs local in-notebook smoke test with sample leaf image.
- **CELL 10**: Launches FastAPI server in background on port `8000`.
- **CELL 11**: Starts Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8000`).
- **CELL 12**: Prints public URL (e.g. `https://xxxx.trycloudflare.com`) and runs live curl verification.

> **Dynamic Tunnel Note:** Cloudflare Tunnel assigns a temporary secure public URL. When a Kaggle session stops or restarts, a new URL is generated.

---

## Google Colab Deployment Guide

1. Open Google Colab and set runtime: **Runtime -> Change runtime type -> T4 GPU**.
2. Open [`notebooks/colab_deployment.ipynb`](file:///c:/Projects/Sage-backend/notebooks/colab_deployment.ipynb).
3. Set your secret token in **Secrets (Key icon on left) -> `HF_TOKEN`**.
4. Run all cells. The Cloudflare Tunnel URL will be printed at the end.

---

## REST API Documentation & Endpoints

Interactive Swagger UI: `https://YOUR-TUNNEL-URL/docs`  
OpenAPI Specification: `https://YOUR-TUNNEL-URL/openapi.json`

### Endpoints Overview

| Method | Endpoint | Description | Auth |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health, device type, base model status | Public |
| `GET` | `/api/v1/crops` | List of supported crops, checkpoints, status | Optional API Key |
| `GET` | `/api/v1/model/status` | Active base model, active adapter, VRAM usage | Optional API Key |
| `POST` | `/api/v1/predict/{crop}` | Diagnose plant leaf from image (crop in path) | Optional API Key |
| `POST` | `/api/v1/predict` | Diagnose plant leaf from image (crop in form) | Optional API Key |
| `POST` | `/api/v1/predict/batch` | Batch sequential diagnosis for multiple images | Optional API Key |

---

## Working cURL Examples

Replace `https://YOUR-TUNNEL-URL` with your live Cloudflare Tunnel URL.

### 1. Health Check
```bash
curl -X GET "https://YOUR-TUNNEL-URL/health"
```
**Response:**
```json
{
  "status": "healthy",
  "service": "SAGE API",
  "device": "cuda",
  "base_model_loaded": true,
  "active_crop": "tomato"
}
```

### 2. List Available Crops
```bash
curl -X GET "https://YOUR-TUNNEL-URL/api/v1/crops"
```

### 3. Diagnose Crop via URL Path (`/predict/{crop}`)
```bash
curl -X POST "https://YOUR-TUNNEL-URL/api/v1/predict/tomato" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@tomato_early_blight.jpg"
```
**Response:**
```json
{
  "success": true,
  "crop": "tomato",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "checkpoint": "tomato/checkpoints/checkpoint-step-600",
  "prediction": {
    "class_name": "Early Blight",
    "crop": "Tomato",
    "disease_type": "Fungal",
    "plant_organ": "Foliar",
    "visual_symptoms": "Concentric dark brown target-like rings with chlorotic halo around leaf spots.",
    "pathogen": "Alternaria solani",
    "matched_taxonomy_class": "Early_Blight",
    "is_canonical_match": true
  },
  "raw_output": "Crop: Tomato\nDiagnosis: Early Blight\nDisease type: Fungal\nPlant organ: Foliar\nVisual symptoms: Concentric dark brown target-like rings with chlorotic halo around leaf spots.\nPathogen: Alternaria solani",
  "inference_time_seconds": 1.84,
  "image_metadata": {
    "width": 1024,
    "height": 768,
    "original_format": "JPEG",
    "channels": 3
  }
}
```

### 4. Diagnose Crop via Form Field (`/predict`)
```bash
curl -X POST "https://YOUR-TUNNEL-URL/api/v1/predict" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "crop=corn" \
  -F "image=@corn_leaf.jpg"
```

### 5. Sequential Batch Diagnosis (`/predict/batch`)
```bash
curl -X POST "https://YOUR-TUNNEL-URL/api/v1/predict/batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "crop=rice" \
  -F "images=@rice_leaf_01.jpg" \
  -F "images=@rice_leaf_02.jpg"
```

---

## n8n Workflow Integration

To integrate SAGE into an autonomous agricultural pipeline in **n8n**:

1. **Add an HTTP Request Node**:
   - **Method**: `POST`
   - **URL**: `https://YOUR-TUNNEL-URL/api/v1/predict/tomato`
   - **Authentication**: Generic Credential Type -> Header Auth:
     - Name: `Authorization`
     - Value: `Bearer {{ $node["Config"].json["SAGE_API_KEY"] }}`
   - **Send Body**: `True`
   - **Body Content Type**: `Form-Data (multipart/form-data)`
   - **Form Fields**:
     - Mode: `n8n Form Data`
     - Parameter Type: `Binary Data`
     - Parameter Name: `image`
     - Input Data Field Name: `data`
2. **Add a Code / Switch Node**:
   - Route downstream alerts (e.g. WhatsApp / Email / Telegram to farmers) when `json.prediction.class_name != "Healthy"`.

---

## External App Integration (Python & JavaScript)

### Python Example
```python
import requests

API_URL = "https://YOUR-TUNNEL-URL/api/v1/predict/tomato"
HEADERS = {"Authorization": "Bearer YOUR_API_KEY"}

with open("sample_leaf.jpg", "rb") as img:
    response = requests.post(API_URL, headers=HEADERS, files={"image": img})

data = response.json()
print("Diagnosis:", data["prediction"]["class_name"])
print("Pathogen:", data["prediction"]["pathogen"])
print("Latency:", data["inference_time_seconds"], "s")
```

### JavaScript / TypeScript Example
```typescript
const formData = new FormData();
formData.append("image", fileInput.files[0]);

const response = await fetch("https://YOUR-TUNNEL-URL/api/v1/predict/soybean", {
  method: "POST",
  headers: {
    "Authorization": "Bearer YOUR_API_KEY",
  },
  body: formData,
});

const result = await response.json();
console.log(result.prediction);
```

---

## GPU Memory & Performance Optimization

| Metric | Tesla T4 (15 GB VRAM) | P100 (16 GB VRAM) |
| :--- | :--- | :--- |
| **Base Model (FP16/BF16)** | ~6.8 GB | ~6.8 GB |
| **Single Crop Adapter** | ~0.15 GB | ~0.15 GB |
| **Inference Activation Peak** | ~0.8 – 1.2 GB | ~0.8 – 1.2 GB |
| **Total VRAM Headroom** | **~6.8 GB Free** | **~7.8 GB Free** |
| **Average Latency** | 1.8 – 2.5 seconds | 1.4 – 2.0 seconds |

---

## Troubleshooting & Common Errors

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `CUDA out of memory` | VRAM exhausted during large batch generation | Keep `ADAPTER_MODE=single`. Images in batch requests are automatically processed sequentially. |
| `Repository not found or access denied` | Private Hugging Face repo without valid token | Set `HF_TOKEN` in Kaggle Secrets or `.env`. Ensure token has Read permissions. |
| `Unsupported image type` | Uploaded image format is not supported | Supported formats: JPEG, PNG, WebP, BMP, TIFF. |
| `413 Request Entity Too Large` | Image file exceeds `MAX_IMAGE_SIZE_MB` (default: 10MB) | Compress image or increase `MAX_IMAGE_SIZE_MB` in `.env`. |
| `Tunnel connection refused / 502` | Server has not finished loading base model | Check `/health` endpoint to ensure `base_model_loaded: true` before sending prediction traffic. |
