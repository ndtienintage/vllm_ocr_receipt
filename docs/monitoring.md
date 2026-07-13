# Monitoring Guide — Receipt OCR vLLM Stack

Hệ thống cung cấp hai lớp giám sát:
1. **Log-based** (mặc định, không cần cài thêm) — structured logs từ pipeline stages
2. **Metrics-based** (tùy chọn) — Prometheus scrape vLLM `:8001/metrics` + Grafana

---

## Lớp 1 — Log Monitoring (mặc định)

### Xem logs pipeline

```bash
# Real-time log của api-ocr
docker logs -f receipt_ocr_api --tail=100

# Chỉ log lỗi
docker logs receipt_ocr_api 2>&1 | grep -E "ERROR|WARN|CRITICAL"

# vLLM — chỉ errors quan trọng
docker logs receipt_ocr_vllm 2>&1 | grep -E "ERROR|CRITICAL"
```

### Cấu trúc log pipeline

Mỗi request có `ref` (từ `reference_id` của caller hoặc `auto-<hex>`) xuyên suốt các bước:

```
[ref=req-001] REQ RECEIVED  | source=url
[ref=req-001] IMG DECODED   | 0.12s | imgs=1
[ref=req-001] PROCESSING    | queue_wait=1.23s
[ref=req-001] STAGE preprocess | 0.45s | imgs=1 | valid=1
[ref=req-001] LLM attempt 1/1  | 4.32s | tokens=1821 (p=1654 c=167) | visual≈1500 | stop=stop
[ref=req-001] STAGE llm_vision | 4.35s | imgs=1
[ref=req-001] STAGE postprocess | 0.01s
[ref=req-001] DONE          | queue_wait=1.23s | processing=4.91s | total=6.14s | imgs=1 | status=success
```

### Metrics quan trọng từ logs

| Log field | Ý nghĩa | Ngưỡng cần chú ý |
|---|---|---|
| `queue_wait` | Thời gian chờ slot semaphore | > 30s → vLLM quá tải |
| `processing` (DONE) | Thời gian thực thi pipeline | > 60s → ảnh phức tạp hoặc vLLM chậm |
| `STAGE preprocess` | PaddleOCR + resize | Thường < 1s |
| `STAGE llm_vision` | Thời gian inference Qwen3-VL | Thường 3–15s tùy độ phức tạp |
| `visual≈` | Ước lượng visual tokens | > 1800 → ảnh lớn, tốn VRAM |
| `stop=length` | LLM bị truncate | Xem `VLLM_MAX_TOKENS` |
| `CANCELLED` | Request timeout hoặc client disconnect | Bình thường nếu ít |

### Health endpoints

```bash
# Liveness — FastAPI + vLLM còn alive
curl http://localhost:8000/health
# → {"status":"healthy","version":"2.1.0","components":{"vllm":"healthy"}}
# HTTP 503 nếu vLLM không phản hồi

# Readiness — xác nhận GPU + vision encoder + LLM thực sự chạy được
# Gửi 1 request generate 1 token với ảnh 1×1 px; kết quả cache 5s
curl http://localhost:8000/ready
# → {"status":"ready","cached":false}
# HTTP 503 nếu GPU/model stack có vấn đề
```

> Docker healthcheck của `api-ocr` dùng `/ready` (interval 30s, start_period 60s).

---

## Lớp 2 — Metrics (tùy chọn: Prometheus + Grafana)

### Nguồn metrics

| Nguồn | Endpoint | Nội dung |
|---|---|---|
| vLLM built-in | `http://localhost:8001/metrics` | Inference, queue, KV cache, tokens, latency |

```bash
# Kiểm tra metrics vLLM
curl http://localhost:8001/metrics | grep -E "^vllm:" | head -30
```

### Metrics vLLM quan trọng

#### Throughput

```promql
sum(rate(vllm:generation_tokens_total{job="vllm-server"}[1m]))
```

- **Qwen3-VL-8B-FP8** trên L4 24GB: kỳ vọng **80–150 tokens/s** tùy batch size và độ phức tạp ảnh
- Giảm đột ngột → vLLM overload, VRAM đầy, hoặc model bị OOM

#### Time to First Token (TTFT)

```promql
histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))
```

| p95 TTFT | Trạng thái |
|---|---|
| < 2s | Tốt |
| 2–5s | Chấp nhận được (ảnh receipt nhiều visual tokens) |
| > 5s | Queue sâu hoặc KV cache đầy — cần kiểm tra |

#### Time per Output Token (TPOT)

```promql
histogram_quantile(0.95, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le))
```

- **< 30ms**: Tốt
- **> 100ms**: GPU utilization thấp hoặc memory pressure

#### Queue Depth

```promql
vllm:num_requests_waiting    # chờ trong queue vLLM
vllm:num_requests_running    # đang GPU xử lý
vllm:num_requests_swapped    # bị swap ra CPU (nguy hiểm)
```

- `waiting > 10` → đang có hàng đợi, bình thường trong burst
- `swapped > 0` → VRAM quá tải, latency sẽ tăng vọt

#### KV Cache Utilization

```promql
vllm:gpu_cache_usage_perc * 100
```

| % | Trạng thái |
|---|---|
| < 80% | An toàn |
| 80–90% | Cảnh báo — request mới phải chờ cache giải phóng |
| > 90% | Nguy hiểm — vLLM swap + preempt, latency đột biến |

### Cài đặt Prometheus + Grafana (tùy chọn)

Tạo file `docker/docker-compose.monitoring.yml`:

```yaml
services:
  prometheus:
    image: prom/prometheus:v3.10.0
    container_name: receipt_ocr_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --web.enable-lifecycle
    restart: unless-stopped

  grafana:
    image: grafana/grafana:12.0-ubuntu
    container_name: receipt_ocr_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

Tạo `docker/monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: vllm-server
    static_configs:
      - targets: ['host.docker.internal:8001']
```

Khởi động:

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

### Alert Rules tham khảo

| Alert | Điều kiện | Pending | Hành động |
|---|---|---|---|
| `VLLM_DOWN` | `up{job="vllm-server"} == 0` | 1m | `docker restart receipt_ocr_vllm` |
| `KV_CACHE_FULL` | KV cache > 90% | 2m | Giảm `--max-num-seqs` hoặc load |
| `QUEUE_DEEP` | `num_requests_waiting > 15` | 3m | Giảm concurrent request rate |
| `REQUESTS_SWAPPING` | `num_requests_swapped > 0` | 1m | Giảm tải ngay lập tức |
| `TTFT_SPIKE` | TTFT p95 > 5s | 5m | Kiểm tra KV cache + queue depth |

---

## Kiểm tra nhanh tình trạng hệ thống

```bash
#!/bin/bash
# health-check.sh — chạy bất kỳ lúc nào để kiểm tra

echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

echo ""
echo "=== Containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=== API Health ==="
curl -sf http://localhost:8000/health | python3 -m json.tool
echo ""
curl -sf http://localhost:8000/ready | python3 -m json.tool

echo ""
echo "=== vLLM Queue ==="
curl -sf http://localhost:8001/metrics \
  | grep -E "^vllm:(num_requests|gpu_cache_usage)" \
  | grep -v "^#"

echo ""
echo "=== Recent Errors ==="
docker logs receipt_ocr_api --since=1h 2>&1 | grep -c "ERROR" | xargs -I{} echo "api-ocr errors (1h): {}"
docker logs receipt_ocr_vllm --since=1h 2>&1 | grep -c "ERROR" | xargs -I{} echo "vllm errors (1h): {}"
```

---

## Troubleshooting phổ biến

### Prometheus không scrape được vLLM

```bash
# Kiểm tra target status
curl http://localhost:9090/api/v1/targets | python3 -m json.tool | grep -A5 "vllm"

# Test trực tiếp
curl http://localhost:8001/metrics | head -20
```

### Grafana không hiện data

1. **Connections → Data Sources → Prometheus → Test**
2. URL phải là `http://prometheus:9090` (container name, không phải `localhost`)
3. Chọn time range "Last 1 hour" để có data gần đây
