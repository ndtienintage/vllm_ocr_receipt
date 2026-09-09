"""CPU-only regressions; run with python -m unittest on this module.

Kept under eval because this repository ignores the entire tests/ directory.
These verify preprocessing invariants, not VLM extraction accuracy.
"""

import unittest

import numpy as np

from src.preprocessing import budget
from src.preprocessing.config import load_config
from src.preprocessing.contracts import PipelineContext, Purpose, StageReport
from src.preprocessing.stages.fit import FitStage


class ResolutionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(env={})

    def context(self, width, height, text_height, purpose=Purpose.VLM):
        ctx = PipelineContext(
            image=np.full((height, width, 3), 255, dtype=np.uint8),
            config=self.config, services=None, purpose=purpose,
        )
        ctx.meta["median_text_height_px"] = text_height
        return ctx

    def test_tall_receipt_reports_lost_legibility_with_zoom_off(self):
        plan = budget.solve(1500, 12000, 24.0, self.config.fit)
        self.assertFalse(plan.zoom_applied)
        self.assertLess(plan.text_h_out, 8.0)
        self.assertTrue(plan.legibility_capped)

    def test_unknown_text_height_is_not_asserted_unreadable(self):
        plan = budget.solve(1500, 12000, None, self.config.fit)
        self.assertIsNone(plan.text_h_out)
        self.assertFalse(plan.legibility_capped)

    def test_fallback_preserves_source_pixels(self):
        ctx = self.context(1500, 12000, 24.0, Purpose.FALLBACK_OCR)
        source = ctx.image
        report = StageReport("fit")
        FitStage().run(ctx, report)
        self.assertIs(ctx.image, source)
        self.assertEqual(ctx.image.shape, (12000, 1500, 3))
        self.assertEqual(report.skipped_reason, "preserve_source_resolution_for_ocr")

    def test_tall_fit_reports_effective_text_size(self):
        ctx = self.context(1500, 12000, 24.0)
        FitStage().run(ctx, StageReport("fit"))
        self.assertLess(ctx.meta["text_h_at_output_px"], 8.0)
        self.assertTrue(ctx.meta["legibility_capped"])
        h, w = ctx.image.shape[:2]
        self.assertLessEqual(h * w, self.config.fit.max_pixels)

    def test_alignment_cannot_hide_loss_in_weaker_axis(self):
        self.config.fit.min_pixels = 0
        self.config.fit.max_side = 320
        ctx = self.context(140, 1000, 48.0)
        FitStage().run(ctx, StageReport("fit"))
        scale = min(ctx.image.shape[1] / 140, ctx.image.shape[0] / 1000)
        self.assertEqual(ctx.meta["text_h_at_output_px"], round(48 * scale, 1))
        self.assertTrue(ctx.meta["legibility_capped"])

    def test_quality_distinguishes_source_from_fitted_image(self):
        from src.extraction.processing import _build_image_quality
        result = _build_image_quality([{
            "median_text_height_px": 48.0,
            "text_h_at_output_px": 7.2,
            "legibility_capped": True,
        }], ref="regression")
        self.assertFalse(result["low_legibility"])
        self.assertTrue(result["output_low_legibility"])
        self.assertTrue(result["legibility_capped"])

    def test_quality_unknown_is_not_false(self):
        from src.extraction.processing import _build_image_quality
        result = _build_image_quality([{}], ref="regression")
        self.assertIsNone(result["text_h_at_output_px"])
        self.assertIsNone(result["output_low_legibility"])
        self.assertIsNone(result["legibility_capped"])

    def test_quality_partial_measurement_is_not_a_clean_bill_of_health(self):
        from src.extraction.processing import _build_image_quality
        result = _build_image_quality([
            {"text_h_at_output_px": 48.0, "legibility_capped": False}, {},
        ], ref="regression")
        self.assertIsNone(result["output_low_legibility"])
        self.assertIsNone(result["legibility_capped"])


if __name__ == "__main__":
    unittest.main()
