# Deployment Guide — Receipt OCR vLLM Stack

## Tổng quan kiến trúc

```
              ┌──────────────────────────────────────┐
  Client      │          Ubuntu 22.04 Server          │
 ──────►      │  ┌──────────────┐  ┌──────────────┐  │
 POST /api/   │  │   api-ocr    │  │ vllm-server  │  │
 ocr/extract  │  │  FastAPI     │─►│  Qwen3-VL    │  │
              │  │  :8000       │  │  :8001       │  │
              │  └──────────────┘  └──────────────┘  │
              │       receipt_ocr_network (bridge)     │
              └──────────────────────────────────────┘
```

### Pipeline xử lý (bên trong api-ocr)

```
Client Request
    │
    ▼
asyncio.wait_for (REQUEST_TIMEOUT=300s → HTTP 408)
    │
    ▼
asyncio.Semaphore.acquire() [capacity = GLOBAL_CONCURRENCY=24]
    │
    ├─ preprocess   PaddleOCR det → orientation → crop → resize
    ├─ llm_vision   Qwen3-VL-8B-FP8 via vLLM (guided JSON decoding)
    └─ postprocess  CJK strip → summary filter → item validate
```

---

## Phiên bản các thành phần

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| **Ubuntu** | 22.04 LTS | Bắt buộc (driver compatibility) |
| **Docker Engine** | ≥ 26.x | Cài qua get.docker.com |
| **Docker Compose** | ≥ 2.27 (plugin) | Đi kèm Docker Engine |
| **NVIDIA Driver** | ≥ 535 | CUDA 12.x |
| **NVIDIA Container Toolkit** | ≥ 1.17 | GPU access cho Docker |
| **Python** | 3.11 (trong Docker) | Pin trong dockerfile |
| **vLLM** | v0.19.1-ubuntu2404 | `vllm/vllm-openai:v0.19.1-ubuntu2404` |
| **Model** | `Qwen/Qwen3-VL-8B-Instruct-FP8` | ~8.5 GB HuggingFace cache |
| **FastAPI** | 0.115.6 | Xem requirements.txt |
| **Uvicorn** | 0.41.0 + uvloop 0.19.0 | ASGI server |
| **OpenAI SDK** | 1.82.0 | Client gọi vLLM API |
| **PaddleOCR** | 3.2.0 | Text detection + orientation |

---

## Yêu cầu phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| **CPU** | 8 Cores | 16 Cores |
| **RAM** | 32 GB | 48 GB |
| **GPU** | **NVIDIA L4 (24 GB VRAM)** | L4 |
| **Disk** | 80 GB SSD | 200 GB NVMe |
| **Network** | 100 Mbps | 1 Gbps |

### Phân bổ tài nguyên

| Container | CPU Limit | RAM Limit | VRAM |
|---|---|---|---|
| `receipt_ocr_vllm` | 8 cores | 16 GB | ~16.8 GB (0.7 × 24 GB) |
| `receipt_ocr_api` | 4 cores | 16 GB | ~2 GB (PaddleOCR) |

**VRAM tổng:** ~18.8 GB / 24 GB → an toàn.

> `vllm-server` dùng thêm `shm_size: 8g` (shared memory cho KV cache) và `--mm-processor-cache-gb 2` (cache preprocessed image qua shared memory — giảm CPU overhead khi retry).

---

## Bước 1 — Chuẩn bị server Ubuntu 22.04

### 1.1 Cập nhật hệ thống

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential ca-certificates gnupg lsb-release
```

### 1.2 Cài NVIDIA Driver 535

```bash
# Kiểm tra GPU nhận được chưa
lspci | grep -i nvidia

# Cách A — tự động
sudo ubuntu-drivers autoinstall

# Cách B — chỉ định version
sudo apt install -y nvidia-driver-535

# REBOOT bắt buộc
sudo reboot
```

```bash
# Sau reboot
nvidia-smi
```

### 1.3 Cài Docker Engine (≥ 26.x)

```bash
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

docker version           # Engine ≥ 26.x
docker compose version   # Compose ≥ 2.27
```

### 1.4 Cài NVIDIA Container Toolkit (bắt buộc)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Kiểm tra — phải thấy GPU output
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## Bước 2 — Lấy mã nguồn và cấu hình

### 2.1 Clone project

```bash
sudo mkdir -p /opt/receipt_ocr
sudo chown $USER:$USER /opt/receipt_ocr
git clone https://github.com/PoiName1923/receipt_ocr_mapping.git /opt/receipt_ocr
cd /opt/receipt_ocr
```

### 2.2 Tạo file `.env`

```bash
cp .env.example .env
nano .env
```

Điền các giá trị thực tế:

```env
# ── HuggingFace (cần cho lần tải model đầu tiên) ────────────────
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Các biến quan trọng khác (giữ default từ `.env.example` nếu không cần thay đổi):

```env
# vLLM
VLLM_BASE_URL=http://vllm-server:8001/v1
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8
VLLM_MAX_TOKENS=2560
VLLM_TEMPERATURE=0.1

# Timeout — 1 lớp duy nhất bao toàn bộ vòng đời request (queue wait + xử lý)
REQUEST_TIMEOUT=300

# Concurrency — phải khớp --max-num-seqs của vLLM (hoặc nhỏ hơn)
GLOBAL_CONCURRENCY=24
```

### 2.3 Cấu trúc project

```
/opt/receipt_ocr/
├── docker/
│   ├── docker-compose.yml       # App stack: vllm-server + api-ocr
│   └── dockerfile               # Build image api-ocr (python:3.11-slim)
├── config/
│   └── exclude_item_patterns.txt  # Từ khóa loại bỏ dòng item không cần thiết
├── src/                         # FastAPI application source
│   ├── main.py                  # Lifespan (warmup vLLM client)
│   ├── core/config.py           # Cấu hình tập trung (single source of truth)
│   ├── api/                     # schema.py, receipt.py, server.py
│   ├── pipeline/                # preprocessing, chunked_extractor, llm_extractor, postprocessor
│   ├── models/                  # llm_client.py (VLLMClient)
│   └── utils/                   # logging_utils, errors, image_utils
├── requirements.txt
├── .env                         # Secrets — KHÔNG commit lên git
└── .env.example                 # Template cho người dùng mới
```

---

## Bước 3 — Khởi động hệ thống

```bash
cd /opt/receipt_ocr/docker

# Lần ĐẦU: vLLM tải model từ HuggingFace (~5-15 phút tùy tốc độ mạng)
docker compose up -d

# Theo dõi tiến trình khởi động
docker logs -f receipt_ocr_vllm 2>&1 | grep -E "Loading|Loaded|startup complete|ERROR"
docker logs -f receipt_ocr_api --tail=50
```

**Dấu hiệu vLLM sẵn sàng:** log in `"Application startup complete"` (thường sau 2-5 phút khi model đã cache).

**Dấu hiệu api-ocr sẵn sàng:** log in `"vLLM client initialized"`.

> `api-ocr` chờ `vllm-server` healthy trước khi start (Docker `depends_on: condition: service_healthy`).

### Kiểm tra toàn bộ stack

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Liveness — FastAPI + vLLM còn sống
curl -sf http://localhost:8000/health && echo "✅ FastAPI OK"

# Readiness — GPU + vision encoder + LLM đang hoạt động thực sự
curl -sf http://localhost:8000/ready && echo "✅ GPU ready OK"

# vLLM trực tiếp
curl -sf http://localhost:8001/health && echo "✅ vLLM OK"
```

---

## Bước 4 — Kiểm thử API

### 4.1 Health endpoints

```bash
# Liveness (HTTP 200 = API + vLLM sống)
curl http://localhost:8000/health
# → {"status":"healthy","version":"2.1.0","components":{"vllm":"healthy"}}

# Readiness (HTTP 200 = GPU stack hoàn toàn sẵn sàng)
curl http://localhost:8000/ready
# → {"status":"ready","cached":false}
```

### 4.2 OCR với ảnh URL

```bash
curl -X POST http://localhost:8000/api/ocr/extract \
  -H "Content-Type: application/json" \
  -d '{
    "images_url": ["https://example.com/receipt.jpg"],
    "reference_id": "req-001"
  }'
```

> `images_url` phải là danh sách chứa **đúng 1 URL**. Truyền 0 hoặc nhiều hơn 1 → HTTP 422.

### 4.3 OCR với ảnh Base64

```bash
IMAGE_B64=$(base64 -w 0 /path/to/receipt.jpg)

curl -X POST http://localhost:8000/api/ocr/extract \
  -H "Content-Type: application/json" \
  -d "{
    \"images_base64\": [\"$IMAGE_B64\"],
    \"reference_id\": \"req-002\"
  }"
```

### 4.4 Kết quả mong đợi

```json
{
  "merchant_name": "Chuỗi siêu thị ABC",
  "merchant_address": "123 Nguyễn Huệ, Q.1, TP.HCM",
  "transaction_date": "2024-01-15",
  "transaction_time": "14:30",
  "receipt_code": "HD-001234",
  "currency": "VND",
  "payment_method": "CASH",
  "items": [
    {"name": "Sữa tươi", "quantity": 2.0, "price": 35000, "total": 70000}
  ],
  "subtotal": 70000,
  "total_amount": 70000
}
```

---

## Cấu hình Firewall (UFW)

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH (bắt buộc — tránh tự khóa)
sudo ufw allow 22/tcp

# API công khai
sudo ufw allow 8000/tcp     # FastAPI

# vLLM — KHÔNG expose ra ngoài (chỉ dùng qua Docker network nội bộ)
# Port 8001 không cần mở UFW

sudo ufw enable
sudo ufw status verbose
```

---

## Vận hành hàng ngày

### Khởi động lại toàn bộ stack

```bash
cd /opt/receipt_ocr/docker
docker compose down
docker compose up -d
```

### Restart service đơn lẻ

```bash
# Restart vLLM (sau khi đổi model args trong docker-compose.yml)
docker compose restart vllm-server

# Restart FastAPI (sau khi update code và rebuild)
docker compose build api-ocr
docker compose up -d --no-deps api-ocr
```

### Xem logs

```bash
# Pipeline timing (queue_wait, stage latency)
docker logs -f receipt_ocr_api --tail=100

# vLLM — chỉ errors
docker logs -f receipt_ocr_vllm 2>&1 | grep -E "ERROR|CRITICAL"
```

### Update code không downtime

```bash
cd /opt/receipt_ocr
git pull origin main

cd docker
docker compose build api-ocr
docker compose up -d --no-deps api-ocr
```

---

## Auto-start khi server reboot

```bash
cat > /etc/systemd/system/receipt-ocr.service << 'EOF'
[Unit]
Description=Receipt OCR Stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/receipt_ocr/docker
ExecStart=docker compose up -d
ExecStop=docker compose down
User=root
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable receipt-ocr.service
sudo systemctl start receipt-ocr.service
sudo systemctl status receipt-ocr.service
```

---

## Backup

```bash
cat > /opt/backup-ocr.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/receipt_ocr/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# HuggingFace model cache (Qwen3-VL)
docker run --rm \
  -v receipt_ocr_huggingface-cache:/data \
  -v "$BACKUP_DIR":/backup \
  alpine tar czf /backup/huggingface-cache.tar.gz /data 2>/dev/null

# PaddleOCR model cache
docker run --rm \
  -v receipt_ocr_paddleocr-models:/data \
  -v "$BACKUP_DIR":/backup \
  alpine tar czf /backup/paddleocr-models.tar.gz /data 2>/dev/null

# Xóa backup cũ hơn 7 ngày
find /opt/backups/receipt_ocr -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
echo "Backup done: $BACKUP_DIR"
EOF

chmod +x /opt/backup-ocr.sh
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/backup-ocr.sh >> /var/log/ocr-backup.log 2>&1") | crontab -
```

---

## Troubleshooting

### vLLM không khởi động

```bash
docker logs receipt_ocr_vllm 2>&1 | tail -50
```

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `CUDA out of memory` | VRAM không đủ | Giảm `--gpu-memory-utilization 0.65` trong docker-compose.yml |
| `No CUDA GPUs available` | Thiếu NVIDIA Container Toolkit | Chạy lại bước 1.4 |
| `Token limit exceeded` | `max-model-len` quá lớn | Giảm `--max-model-len 4096` |
| `Connection refused` | vLLM chưa ready | Đợi thêm (lần đầu tải model ~15 phút) |

### api-ocr báo 503 khi gọi OCR

```bash
# Kiểm tra vLLM còn sống
curl http://localhost:8001/health

# Kiểm tra GPU stack thực sự ready
curl http://localhost:8000/ready

# Log pipeline
docker logs receipt_ocr_api 2>&1 | grep -E "ERROR|WARN|upstream"
```

### Request timeout (HTTP 408)

Kiểm tra xem bottleneck ở đâu:

```bash
# Xem queue_wait vs processing time
docker logs receipt_ocr_api 2>&1 | grep "DONE\|CANCELLED" | tail -20
```

- `queue_wait` cao → vLLM quá tải, tăng `GLOBAL_CONCURRENCY` hoặc giảm request rate
- `processing` cao → ảnh phức tạp hoặc VRAM không đủ, xem `nvidia-smi`
- Tăng `REQUEST_TIMEOUT` trong `.env` nếu inference chậm hợp lý

### api-ocr không kết nối được vLLM

```bash
# Kiểm tra cùng network
docker network inspect receipt_ocr_receipt_ocr_network
docker exec receipt_ocr_api curl -sf http://vllm-server:8001/health
```

---

## Checklist triển khai

- [ ] Ubuntu 22.04 LTS đã cài xong
- [ ] NVIDIA Driver ≥ 535 — `nvidia-smi` hoạt động
- [ ] Docker Engine ≥ 26 — `docker version` OK
- [ ] NVIDIA Container Toolkit — `docker run --gpus all nvidia/cuda:12.0... nvidia-smi` OK
- [ ] File `.env` đã tạo với `HUGGING_FACE_HUB_TOKEN`
- [ ] `docker compose up -d` thành công
- [ ] `docker logs receipt_ocr_vllm` in `"Application startup complete"`
- [ ] `curl http://localhost:8000/health` → `{"status":"healthy",...}`
- [ ] `curl http://localhost:8000/ready` → `{"status":"ready",...}`
- [ ] Test OCR với ảnh thực tế → kết quả JSON hợp lệ
- [ ] Firewall UFW đã cấu hình (port 22, 8000)
- [ ] systemd service đã enable (auto-start sau reboot)
- [ ] Cron backup đã cài
