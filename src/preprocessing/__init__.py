"""preprocessing — Bộ xử lý ảnh hoá đơn CHÍNH, consumer là
Qwen3-VL-8B-Instruct. Stage-based, config-driven, adaptive.

Tài liệu thiết kế: .claude/pharse/output/phase1_design.md (kiến trúc + ADR).

Seam tích hợp với orchestrator (`src.extraction.processing`):
    from src.preprocessing.compat import preprocess_image_with_meta
"""
