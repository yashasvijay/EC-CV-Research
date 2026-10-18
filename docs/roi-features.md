# Local feature reseeding

> Current scope (2026-10-18): Both paired selector and updated profile measurements are complete; results and caveats are in reports/roi-features/results.md. The async timing result is a separate empty-track workload. See [verified performance](verified-performance.md).

`src/features.py` provides `select_roi_features()` for synchronous LK reseeding and the asynchronous detection worker. The tracker previously passed a full grayscale frame with a bounding-box mask. That mask restricted selected points, but OpenCV still computed the corner-response image over the supplied image before applying the mask. The [OpenCV implementation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/featureselect.cpp) shows this ordering.

The shared selector now clips each active xyxy box to the image, crops the grayscale image to its integer envelope, creates a matching **local** uint8 mask, and selects corners on that crop. The mask excludes pixel centers below fractional lower bounds; upper bounds are exclusive. Returned float32 `(N,1,2)` points are translated into full-frame coordinates and checked against the clipped box. Small, empty, off-screen, or textureless ROIs return an empty point set. Missed tracks are not reseeded. The helper does not modify the input image.

This removes full-image masks and repeated full-image corner-response work from each track refresh. The async worker already cropped its input; it now shares the mask, clipping and coordinate-boundary rules. No detector, association, LK-flow, or scheduling thresholds change. Cropping changes image-border derivative/nonmaximum-suppression behavior, so the selected corner population can differ. Full-sequence HOTA/MOTA preservation is not established by bounds tests or a short timing run.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_features.py tests/test_phase1.py \
  tests/test_async_pipeline.py tests/test_profile_pipeline.py
```

The 1080p test intercepts the OpenCV call to verify that it receives the crop and same-sized local mask, and checks each track's points against its own box. Additional tests cover fractional and image-clipped boxes, degenerate ROIs, no texture, inactive tracks, input immutability, async reuse, and benchmark output. Timing assertions do not require a fixed speedup on arbitrary test hardware.

## Measure the change

The paired benchmark uses the original profile's MOT-format boxes on source frames 1–50. It compares the former full-frame-mask implementation with the new selector on identical boxes and images, alternating their order across frames/repeats. It includes mask creation and output-coordinate processing in each selector's timed interval, excludes decode/logging, and reports per-frame mean/p95/p99, repeat statistics, corner counts, CPU, and sampled RSS. It uses three repetitions and one OpenCV thread by default. This isolates reseeding work; it does not estimate full pipeline FPS or tracking accuracy.

```bash
.venv/bin/python -m scripts.benchmark_roi_features \
  --sequence data/raw/MOT17/train/MOT17-02-FRCNN \
  --tracks reports/profiling/tracks.txt --frames 50 --repeats 3 --threads 1 \
  --output reports/roi-features/paired-repeat

.venv/bin/python -m scripts.profile_pipeline \
  --model models/yolo11n.onnx \
  --source data/raw/MOT17/train/MOT17-02-FRCNN \
  --frames 50 --threads 1 --warmup 10 \
  --output reports/roi-features/profile-repeat
```

Run sequentially after checking for competing work. Outputs must be new directories. The second command measures the updated pipeline, including cProfile overhead; compare its native feature-selection span with the historical profile as a descriptive comparison across separate sessions. Use the contemporaneous paired benchmark for the isolated selector reduction. The old profile in `reports/profiling/` and completed grid-search results are preserved.

`reports/roi-features/preflight.json` records the observed load check, and `historical_artifact_hashes.json` records the existing result files before this experiment. Neither an idle CPU check nor repeat timing establishes controlled thermal conditions. Accuracy metrics remain null pending full-sequence evaluation. The profiler and future grid searches fingerprint the new helper; the old completed grid cannot be resumed against changed tracking code without a configuration mismatch, which is intentional.
