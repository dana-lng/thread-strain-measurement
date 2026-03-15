# AGENTS.md

## Project overview
This repository contains a computer-vision pipeline for measuring yarn/thread strain from high-speed camera images.

The task is to detect two horizontal marker bands on vertically oriented moving threads and compute their vertical distance over time. The primary goal is a robust, explainable, non-deep-learning baseline.

## Main objective
For each image sequence:
1. detect the upper marker band
2. detect the lower marker band
3. compute marker distance in pixels
4. derive relative strain over time
5. export results as plots and CSV files

Relative strain is typically defined as:

strain(t) = (d(t) - d0) / d0

where:
- d(t) = marker distance in frame t
- d0 = reference distance in the initial state

## Preferred approach
Default to classical computer vision and signal processing.

Prioritize:
- grayscale preprocessing
- denoising
- edge/gradient-based analysis
- row-wise signal aggregation
- peak detection
- temporal tracking between frames
- outlier rejection and smoothing

Good baseline strategies include:
- Sobel/Laplacian-based row profiles
- peak detection on 1D vertical profiles
- local window tracking using previous frame positions
- optional Kalman filtering for smoothing

## Explicit non-goals
Do not introduce CNNs, U-Nets, YOLO, or other deep-learning models unless explicitly requested.
Do not add unnecessary training pipelines, annotation tools, or ML dependencies.
Do not overengineer the first solution.

## Coding principles
- Use Python.
- Prefer simple, readable, modular code.
- Add comments where the reasoning is not obvious.
- Keep functions small and testable.
- Avoid hidden magic constants; expose important parameters clearly.
- Preserve reproducibility.

## Preferred libraries
Use these by default when appropriate:
- numpy
- opencv-python
- scipy
- matplotlib
- pandas

Avoid adding heavy dependencies unless clearly justified.

## Data handling
- Never modify raw input images.
- Treat raw data as read-only.
- Write derived outputs to dedicated output folders.
- Keep paths configurable.

## Suggested repository structure
- `src/` for implementation code
- `scripts/` for runnable entry points
- `configs/` for parameters
- `data/raw/` for unmodified input data
- `data/processed/` for derived intermediate data
- `results/` for CSVs, plots, and evaluation outputs
- `tests/` for automated tests
- `notebooks/` only for exploration, not production logic

## Expected outputs
For each sequence, generate:
- detected marker positions per frame
- marker distance per frame
- relative strain per frame
- optional quality/confidence indicators
- CSV export
- diagnostic plots

## Validation expectations
When changing detection logic:
- test on a small subset first
- visualize detections on sample frames
- check for temporal consistency
- flag suspicious jumps or missing detections

## When implementing changes
Codex should:
1. understand the current pipeline before editing
2. prefer minimal targeted changes
3. explain assumptions briefly
4. avoid unrelated refactors
5. update documentation when behavior changes

## Style for analysis
Prefer practical engineering decisions over theoretical complexity.
When multiple options exist, choose the simplest robust solution first.
If detection is unreliable, add diagnostics before changing algorithms.

## File safety
Do not delete user data.
Do not overwrite result files without clear reason.
Do not rename major folders unless explicitly requested.

## Definition of success
A successful solution is:
- understandable
- reproducible
- robust on real image sequences
- easy to inspect visually
- feasible for a student project with limited weekly time