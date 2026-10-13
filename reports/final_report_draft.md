# Adaptive Inference Gating for Edge-Oriented Tracking-by-Detection on Apple Silicon

**Yashas Vijay**  
*Technical report draft · 12 October 2026*

<div style="background:#F9F7F5;color:#2C2A29;border-left:4px solid #A34828;padding:12px 16px;">
<strong>Evidence status.</strong> This draft reports completed initial Python-pipeline experiments and a validated native implementation. The 48-configuration search remains in progress. Native latency measurements, selected-configuration validation, and power measurements are pending. The 50–60% inference-reduction objective is a research target, not an achieved result.
</div>

## Abstract

Repeated neural detection imposes a substantial compute cost on multi-object tracking. We study an adaptive tracking-by-detection pipeline that schedules YOLO11n inference using sparse Lucas–Kanade motion, feature survival, forward–backward consistency, and bounding-box displacement. The research objective is a 50–60% reduction in detector calls without loss of HOTA or MOTA on MOT17, using Apple Silicon as the initial CPU platform. A modular implementation combines ONNX inference, persistent track association, per-frame instrumentation, and offline TrackEval evaluation. In initial single-run experiments, adaptive scheduling reduced detector calls by 19.33% on MOT17-02 and 29.90% on MOT17-04. Throughput increased from 3.84 to 4.21 FPS and from 2.83 to 3.08 FPS, respectively. MOT17-02 HOTA changed from 25.82 to 25.94; adaptive MOT17-04 accuracy remains unevaluated. A C++17 engine implements the motion reductions and gating rules through validated NumPy buffers and pybind11. Its build and numerical checks are complete, while native timing and real-video integration remain to be evaluated. We formalize the method, specify the search and measurement protocol, and provide a reporting framework for accuracy–compute tradeoffs. Current evidence supports modest inference savings on two sequences, but does not establish the target reduction, real-time deployment, or energy savings.

**Keywords:** multi-object tracking; adaptive inference; optical flow; edge systems; Apple Silicon; HOTA.

## 1. Introduction and Motivation

Tracking-by-detection repeatedly executes a detector and associates its predictions across frames. Full-rate detection simplifies recovery from missed observations, but spends inference capacity even when objects move predictably. Between detector updates, sparse image motion can provide a cheaper propagation signal. The scheduling problem is to decide when that signal has become unreliable enough to justify another detector execution.

Skipping inference creates several risks. A propagated box can drift onto background texture, lose its feature support during occlusion, or retain a plausible translation while an object changes scale. New objects may enter outside existing tracked regions. A detector interval therefore controls both computational load and the delay before a previously unseen object can be discovered. A confidence rule complements that interval by reacting to the observed quality of track propagation.

Our system investigates this tradeoff through explicit heuristics rather than a learned gating network. The core question is whether motion-based scheduling can reduce detector calls by 50–60% while retaining the tracking fidelity of an otherwise identical full-inference pipeline. Apple Silicon provides the initial development platform; the present experiments use CPU inference and offline sequence processing. Their throughput remains below the source videos' 30 FPS, so deployment at capture rate is an unresolved systems objective.

The study has three components: a reproducible detector/tracker interface; a measurable scheduling policy with a fixed parameter search; and a native implementation whose numerical behavior and crossing overhead can be evaluated independently. The contribution is an experimental and engineering framework. Claims of superiority over established trackers require additional baselines and held-out experiments.

## 2. Methodology

### 2.1. Detector and Association

We use the COCO-trained YOLO11n detection model, exported to ONNX with batch size one, fixed 640×640 input, FP32 computation, and raw detection output without embedded NMS. The model belongs to the Ultralytics YOLO11 family. [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

The wrapper letterboxes each BGR frame, converts it to RGB, normalizes values to [0,1], and constructs a contiguous NCHW tensor. ONNX Runtime executes the graph through `CPUExecutionProvider`. Postprocessing selects COCO person class 0, applies a score threshold of 0.25 and class-aware NMS at IoU 0.45, and maps boxes back to source-image coordinates. The recorded model checksum is provided in Appendix A.

At each scheduled detector frame, class-aware Hungarian assignment matches detections to propagated tracks using IoU. Matches below 0.3 are disallowed. Unmatched detections initialize new identities; unmatched tracks are suppressed from output and retained for at most two detector updates before removal. Matched tracks receive newly detected boxes and reseeded corner points. There is no appearance embedding or re-identification model.

The principal baseline executes detection on every frame **and retains the same LK propagation and association machinery**. This control isolates detector scheduling. It is not the minimal-cost implementation of detection-only tracking. A separate eight-state Kalman/Hungarian tracker is available for later experiments. Its state includes aspect-ratio velocity and thus differs from the seven-state original SORT implementation; its results are not included in the present initial comparison. [SORT paper](https://arxiv.org/abs/1602.00763), [reference implementation](https://github.com/abewley/sort).

### 2.2. Sparse Lucas–Kanade Motion Estimation

Let $I_{t-1}$ and $I_t$ denote consecutive grayscale images. For a feature centered at $\mathbf{p}$, assume approximately constant brightness and a local translation $\mathbf{v}=(u,v)^\top$ over a window $\Omega(\mathbf{p})$. The local registration objective is

$$
\widehat{\mathbf{v}}
=
\underset{\mathbf{v}}{\operatorname{argmin}}
\sum_{\mathbf{x}\in\Omega(\mathbf{p})}
\left[I_t(\mathbf{x}+\mathbf{v})-I_{t-1}(\mathbf{x})\right]^2.
\tag{1}
$$

A first-order expansion yields the optical-flow constraint

$$
I_x(\mathbf{x})u+I_y(\mathbf{x})v+I_{\Delta t}(\mathbf{x})\approx 0,
\qquad
A\mathbf{v}\approx-\mathbf{b},
\qquad
\widehat{\mathbf{v}}=-(A^\top A)^{-1}A^\top\mathbf{b}.
\tag{2}
$$

Each row of $A$ contains $[I_x,I_y]$, and each entry of $\mathbf{b}$ contains the inter-frame intensity difference $I_{\Delta t}$. The normal-equation expression assumes sufficient local gradient variation; ill-conditioned patches cannot determine a reliable displacement. Iterative updates and an image pyramid extend the local linearization to larger motion. This formulation follows Lucas–Kanade registration; the implementation uses OpenCV's pyramidal solver. [Lucas and Kanade](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf), [OpenCV optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html).

Within each detected box, `goodFeaturesToTrack` selects at most 60 corners with quality level 0.01 and minimum separation 5 pixels. Forward and backward `calcOpticalFlowPyrLK` calls use a 21×21 window and `maxLevel=3`. For feature $j$ of track $i$, let $\widehat{\mathbf{p}}_{i,t}^{\,j}$ be the forward estimate and $\widetilde{\mathbf{p}}_{i,t-1}^{\,j}$ its backward reconstruction. Define

$$
e_{i,t}^{\,j}
=
\left\|\widetilde{\mathbf{p}}_{i,t-1}^{\,j}
-\mathbf{p}_{i,t-1}^{\,j}\right\|_2,
\qquad
\mathcal{V}_{i,t}
=
\left\{j:
\begin{array}{l}
s_{i,t}^{f,j}=s_{i,t}^{b,j}=1,\\
e_{i,t}^{\,j}\leq\tau_{\mathrm{fb}},\\
\widehat{\mathbf{p}}_{i,t}^{\,j}\text{ lies inside the frame}
\end{array}
\right\}.
\tag{3}
$$

Both status indicators must report success, and the error must be finite. We fix $\tau_{\mathrm{fb}}=1.5$ pixels. Fewer than six surviving points invalidate propagation. A nonfinite forward point also invalidates the entire track, matching the current tracker's conservative early failure rule.

### 2.3. Bounding-Box Propagation and Normalized Displacement

Write the previous box as $\mathbf{b}_{i,t-1}=(x_1,y_1,x_2,y_2)$ and its width and height as $w_{i,t-1}$ and $h_{i,t-1}$. Over surviving points, compute a coordinate-wise median translation:

$$
\Delta\mathbf{p}_{i,t}^{\,j}
=\widehat{\mathbf{p}}_{i,t}^{\,j}-\mathbf{p}_{i,t-1}^{\,j},
\qquad
\mathbf{m}_{i,t}
=\operatorname{median}_{j\in\mathcal{V}_{i,t}}
\Delta\mathbf{p}_{i,t}^{\,j},
\qquad
\widehat{\mathbf{b}}_{i,t}
=\mathbf{b}_{i,t-1}+(m_x,m_y,m_x,m_y).
\tag{4}
$$

Clip the propagated box to the image boundary. The normalized displacement is

$$
d_{i,t}
=
\frac{\|\mathbf{m}_{i,t}\|_2}
{\max\!\left(1\ \mathrm{pixel},
\sqrt{w_{i,t-1}^{\,2}+h_{i,t-1}^{\,2}}\right)}.
\tag{5}
$$

This ratio is dimensionless and uses the box diagonal **before** the current translation. It measures motion during one frame transition; it is not cumulative geometric drift. The box model supports translation only. Scale change, rotation, and deformation require detector correction or a richer propagation model.

### 2.4. Confidence Drift and Adaptive Scheduling

For an accepted track with $n_{i,t-1}>0$ input points and at least six survivors, define feature survival, median consistency error, and median motion dispersion as

$$
\rho_{i,t}=\frac{|\mathcal{V}_{i,t}|}{n_{i,t-1}},
\qquad
\overline{e}_{i,t}=\operatorname{median}_{j\in\mathcal{V}_{i,t}} e_{i,t}^{\,j},
\qquad
\sigma_{i,t}=\operatorname{median}_{j\in\mathcal{V}_{i,t}}
\left\|\Delta\mathbf{p}_{i,t}^{\,j}-\mathbf{m}_{i,t}\right\|_2.
\tag{6}
$$

Here $\sigma_{i,t}$ is a median residual magnitude, not a statistical standard deviation. Track quality and cumulative confidence are

$$
q_{i,t}
=\rho_{i,t}
\exp\!\left(-\frac{\overline{e}_{i,t}}{\tau_{\mathrm{fb}}}\right)
\exp\!\left(-\frac{\sigma_{i,t}}{\lambda}\right),
\qquad
c_{i,t}=c_{i,t-1}q_{i,t},
\quad \lambda=3\ \mathrm{pixels}.
\tag{7}
$$

Confidence resets to 1 after a detector update only when the reseeded box contains at least six points; otherwise it resets to 0. Insufficient surviving features or a degenerate clipped box force confidence to 0. Consequently, confidence accumulates evidence of unreliable propagation between detector refreshes. It is a heuristic quality score, not a calibrated probability of correct tracking.

Let $\mathcal{T}_t$ contain visible tracks after propagation, and let $s_t$ be the most recent detector frame before the current decision. For a nonempty track set, define $c_t^{\min}=\min_{i\in\mathcal{T}_t}c_{i,t}$ and $d_t^{\max}=\max_{i\in\mathcal{T}_t}d_{i,t}$. The frame-level detector gate is

$$
g_t=
\begin{cases}
1, & \mathcal{T}_t=\varnothing,\\
1, & t-s_t\geq K,\\
1, & c_t^{\min}<\tau_c,\\
1, & d_t^{\max}>\tau_d,\\
0, & \text{otherwise}.
\end{cases}
\tag{8}
$$

The implementation evaluates conditions in the displayed order to retain an unambiguous trigger label. Equality at $\tau_c$ or $\tau_d$ does not trigger inference; equality at $K$ does. If $g_t=1$, execute detection, associate/reseed tracks, and set the last detector frame to $t$. The first frame triggers because no tracks exist. One unreliable object can trigger a detector execution for the entire frame.

The maximum interval limits discovery delay in frame count, while the confidence and motion rules react to observed propagation quality. The bound does not guarantee real-time wall-clock response when processing falls behind capture rate. The native decision API additionally treats nonfinite or out-of-range metric vectors as invalid motion and requests inference. This defensive behavior is separate from the finite-input rule in Eq. (8).

## 3. Systems Architecture and Native Optimization

### 3.1. Modular Execution

| Component | Implemented role | Principal artifact |
|---|---|---|
| Detector | ONNX inference, letterboxing, class filtering, NMS | `src/detector.py` |
| Sparse tracker | Forward/backward LK, translation, confidence, matching | `src/tracker.py` |
| Pipeline | Detector scheduling and stage timing | `src/pipeline.py` |
| Measurement | Latency/FPS, process CPU/RSS, JSON exports | `src/benchmark.py` |
| Dataset adapter | Frame order, timestamps, metadata validation | `src/datasets/mot17.py` |
| Native core | Motion reductions, normalized displacement, decisions | `csrc/include/gating.hpp`, `csrc/src/gating.cpp` |
| Binding layer | Validated buffers and Python object ownership | `csrc/bindings.cpp` |
| Evaluation | MOT17 preprocessing, HOTA/CLEAR/Identity metrics | `scripts/evaluate_tracking.py` |

**Table 1.** Software boundaries separate detector execution, propagation, scheduling, and measurement. Native optimization targets reductions and scheduling around LK outputs; OpenCV already runs the LK solver in native code.

The C++17 core holds an immutable configuration and exposes single-track assessment, ragged batch assessment, and a stateless frame decision. It computes Eqs. (3)–(8) without a Python or OpenCV dependency. Float32 motion arithmetic follows the tracker's buffers; confidence penalties use double precision. Median reductions use partial selection rather than sorting all points. Batch calls reuse scratch capacity within the call. The engine creates no threads and retains no input buffers between calls.

### 3.2. NumPy Layout and Ownership

The pybind11 interface accepts native-endian, aligned, C-contiguous arrays. Point buffers have dtype float32 and shape $(N,2)$ or $(N,1,2)$; status arrays have dtype uint8 and shape $(N,)$ or $(N,1)$. Single boxes use float32 xyxy arrays. Scheduling confidence/displacement vectors use float64. The binding rejects incompatible dtypes, strides, alignment, or shapes rather than silently converting them. This design uses pybind11's NumPy buffer interface while preserving explicit layout contracts. [NumPy interface documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html).

Inputs are borrowed for the duration of a synchronous call. Python references retain their owners until computation returns. Small results, retained-point indices, and temporary scratch storage own their memory. Zero-copy therefore describes **input access**, not an allocation-free pipeline. The batch API consumes concatenated point arrays, int64 offsets of length $M+1$, $M\times4$ boxes, and $M$ previous confidences. Packing separate per-track arrays into that layout can introduce copies and must be included in performance comparisons.

### 3.3. GIL Release and Integration Status

The bindings release the Python global interpreter lock only around pure C++ computation, then reacquire it before constructing Python results. Immutable configuration and call-local scratch permit independent-buffer concurrent calls. Releasing the GIL enables other Python threads to progress; it does not parallelize the native loop automatically. Shared underlying NumPy storage must not be mutated or resized during a call, even when one view is marked read-only. [pybind11 GIL guidance](https://pybind11.readthedocs.io/en/stable/advanced/misc.html).

The module builds as an ARM64 Python extension using CMake and AppleClang. Standalone C++ and Python binding checks have passed, including numerical parity, malformed-buffer rejection, result ownership, batching, and concurrent calls. Correctness tests establish interface behavior, not a speedup.

An isolated `--use-native-gating` runner routes scheduling decisions to the engine while retaining Python LK reductions and association. A direct pipeline/CLI integration patch is prepared but deliberately unapplied while the grid search uses fingerprinted source files. Full native motion-reduction integration into real video runs remains future work. This separation prevents the ongoing search from mixing implementation revisions.

A local speedup has limited effect if it covers only a small fraction $\alpha$ of total pipeline time. Under an idealized Amdahl model, a native speedup $S_g$ in that fraction gives

$$
S_{\mathrm{pipeline}}
=\frac{1}{(1-\alpha)+\alpha/S_g}.
\tag{9}
$$

Packing, conversion, and synchronization can reduce this bound further. We have not measured $\alpha$ for native-eligible reductions, so Eq. (9) is a design constraint rather than a numerical prediction.

## 4. Experimental Protocol

### 4.1. Data and Platform

We use two MOT17 training sequences from the official archive. MOT17-02 contains people walking through a public square; MOT17-04 depicts a pedestrian street at night from an elevated viewpoint. Both contain 1920×1080 images recorded at 30 FPS. Their lengths are 600 and 1,050 frames. MOT17 provides multiple detector variants for the same videos; we select the FRCNN-suffixed image copy once per sequence and run our own detector. [MOT17 archive](https://motchallenge.net/data/MOT17/), [MOTChallenge benchmark](https://arxiv.org/abs/2010.07548).

| Sequence | Frames | Resolution | Source rate | Initial role |
|---|---:|---|---:|---|
| MOT17-02-FRCNN | 600 | 1920×1080 | 30 FPS | Initial scheduling comparison |
| MOT17-04-FRCNN | 1,050 | 1920×1080 | 30 FPS | Additional scene and development search |

**Table 2.** The current study uses training sequences for development. Neither sequence provides an untouched test set after parameter tuning; final selected-policy claims require additional held-out sequences.

The host is macOS ARM64 on Apple Silicon, with eight logical CPU cores reported by the logger. ONNX Runtime and OpenCV each use one configured compute thread. The exact chip model, power mode, thermal trajectory, and background-load measurements were not archived with the initial runs. These fields must be recorded in the final timing study. CPU percentages use 100% to denote one logical core; sampled RSS measures process residency, not total device RAM.

### 4.2. Accuracy and Performance Metrics

TrackEval evaluates complete prediction files with MOT17 pedestrian/distractor preprocessing enabled. We report HOTA for joint detection/association quality, MOTA for aggregate tracking errors, IDF1 for identity consistency, and ID switches as a count. HOTA is averaged over the evaluator's localization thresholds. HOTA, MOTA, and IDF1 are reported in percentage units; MOTA can be negative. [HOTA paper](https://arxiv.org/abs/2009.07736), [TrackEval](https://github.com/JonathonLuiten/TrackEval).

For $N$ processed frames and gate indicators $g_t$, define detector reduction and measured throughput as

$$
R(\theta)=100\left(1-\frac{1}{N}\sum_{t=1}^{N}g_t\right),
\qquad
\mathrm{FPS}=\frac{N}{T_{\mathrm{wall}}}.
\tag{10}
$$

For clarity, MOTA and IDF1 use

$$
\mathrm{MOTA}=100\left(1-
\frac{\sum_t(\mathrm{FN}_t+\mathrm{FP}_t+\mathrm{IDSW}_t)}
{\sum_t\mathrm{GT}_t}\right),
\qquad
\mathrm{IDF1}=100\frac{2\,\mathrm{IDTP}}
{2\,\mathrm{IDTP}+\mathrm{IDFP}+\mathrm{IDFN}}.
\tag{11}
$$

IDTP/IDFP/IDFN are identity-level matches and errors, not detector classification counts. We delegate metric matching and ignored-region handling to TrackEval rather than approximating them from flow confidence.

Each measured frame exports flow, detector, tracker-update, decoding, and total pipeline latency. We report mean, p50, p95, and p99. Wall throughput includes decoding, processing, output writes, and resource instrumentation, but excludes model loading and ten detector warmup executions. Pipeline latency excludes decoding/export. The logger's `association_ms` timer includes both matching **and corner reseeding**, so it cannot isolate Hungarian matching cost alone. Process CPU time divided by measured wall time provides aggregate CPU utilization. No power sensor or external meter was used.

The initial comparison contains one timing repetition per mode and sequence. Final reporting should use at least five repetitions with balanced/randomized order, stable runtime versions, recorded device conditions, and run-level uncertainty. Frame timings are temporally correlated and must not be treated as independent experimental replicates. Failed or truncated runs are excluded through completion status and contiguous frame-ledger checks.

### 4.3. Hyperparameter Search and Policy Selection

The adaptive design vector is $\theta=(K,\tau_c,\tau_d)$, with

$$
\Theta=
\{10,15,20,30\}
\times\{0.5,0.6,0.7\}
\times\{0.15,0.20,0.25,0.30\},
\qquad |\Theta|=48.
\tag{12}
$$

The search randomizes configuration order with seed 42 and executes each configuration in a fresh process. The current search uses one repetition per configuration, full MOT17-04 frames, and offline TrackEval. Source, model, runtime, and completed-run fingerprints support reproducibility and resume. Aggregate CSV and JSON exports retain inference reduction, latency quantiles, FPS, CPU/RSS, HOTA/MOTA/IDF1, and trigger counts.

The efficiency frontier maximizes $R$ and minimizes mean pipeline latency. The quality frontier adds maximizing HOTA. A configuration dominates another only if it is no worse in every stated objective and strictly better in at least one. The HOTA–reduction figure uses a separate two-dimensional projection; latency-optimal points need not lie on that projected frontier.

To evaluate the stated no-degradation goal, predeclare the feasible set

$$
\Theta_{\mathrm{feasible}}=
\left\{\theta\in\Theta:
\begin{array}{l}
\mathrm{HOTA}(\theta)\geq\mathrm{HOTA}_{\mathrm{base}}-\varepsilon_H,\\
\mathrm{MOTA}(\theta)\geq\mathrm{MOTA}_{\mathrm{base}}-\varepsilon_M
\end{array}
\right\},
\qquad
\theta^*\in\underset{\theta\in\Theta_{\mathrm{feasible}}}{\operatorname{argmax}}R(\theta).
\tag{13}
$$

Set $\varepsilon_H=\varepsilon_M=0$ for a literal no-degradation criterion. Any nonzero practical tolerance must be specified before inspecting held-out outcomes. The search's optional HOTA-loss flag constrains HOTA only; it does not enforce the full two-metric criterion in Eq. (13). MOTA, IDF1, and identity-switch behavior must also be reviewed. The 50–60% target is attained only if a feasible validated configuration reaches that reduction band. A high-reduction point outside the fidelity constraint does not satisfy the objective.

## 5. Results and Discussion Framework

### 5.1. Completed Initial Comparisons

| Sequence | Policy | Detector calls | Reduction (%) | FPS | HOTA | MOTA | IDF1 | IDSW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02 | Full inference + LK | 600 | 0.00 | 3.84 | 25.82 | 20.62 | 25.48 | 136 |
| MOT17-02 | Adaptive + LK | 484 | 19.33 | 4.21 | 25.94 | 20.83 | 25.85 | 112 |
| MOT17-04 | Full inference + LK | 1,050 | 0.00 | 2.83 | 32.81 | 31.63 | 34.77 | 237 |
| MOT17-04 | Adaptive + LK | 736 | 29.90 | 3.08 | Pending | Pending | Pending | Pending |

**Table 3.** Completed initial runs use $K=10$, $\tau_c=0.6$, and $\tau_d=0.15$ for adaptive scheduling. Numbers come from the saved full-sequence summaries. Adaptive MOT17-04 accuracy must be evaluated from its own prediction file; it must not be inferred from a tuned grid configuration or substituted from the baseline. The initial table reports single runs, without timing confidence intervals.

On MOT17-02, throughput increased by 9.56% with 116 fewer detector executions. The HOTA change was +0.12 percentage points after rounding; ID switches decreased from 136 to 112. This sequence supports similar measured fidelity under the initial schedule, while the absolute scores indicate substantial remaining detection/association error. It does not establish a statistically supported improvement across datasets.

On MOT17-04, throughput increased by 8.76% with 314 fewer detector executions. Tracking fidelity for that initial adaptive run is still unknown. Completing its evaluation is required before extending the accuracy statement from MOT17-02 to the second sequence.

| Sequence / policy | Mean pipeline (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Mean CPU (%) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| MOT17-02 / full | 252.56 | 242.29 | 325.35 | 446.63 | 96.22 | 258.50 |
| MOT17-02 / adaptive | 230.31 | 246.51 | 400.62 | 611.07 | 92.23 | 275.59 |
| MOT17-04 / full | 346.33 | 337.64 | 430.00 | 541.43 | 97.15 | 278.08 |
| MOT17-04 / adaptive | 317.11 | 366.39 | 551.76 | 1,008.71 | 92.00 | 260.19 |

**Table 4.** Mean pipeline latency decreased, while p95/p99 increased for both adaptive runs. Memory changed in different directions across the two sequences. These results support a throughput–tail-latency tradeoff, not uniform systems improvement or lower energy use.

### 5.2. Stage Costs and Trigger Behavior

| Sequence / policy | Flow stage (ms/frame) | Detector wrapper (ms/frame) | Matching + reseeding (ms/frame) |
|---|---:|---:|---:|
| MOT17-02 / full | 34.18 | 92.44 | 125.93 |
| MOT17-02 / adaptive | 36.48 | 79.88 | 113.92 |
| MOT17-04 / full | 50.08 | 90.43 | 205.79 |
| MOT17-04 / adaptive | 64.88 | 74.37 | 177.84 |

**Table 5.** Means are calculated over all processed frames from `frames.jsonl`; skipped detector/update stages contribute zero. The flow stage includes grayscale conversion and Python reductions, while the detector wrapper includes preprocessing and postprocessing. Matching and corner reseeding share one timer. Small timing overhead outside these stages accounts for the difference from total pipeline means.

The scheduled update stage is a major measured cost. Native gate reductions therefore cannot be assumed to dominate the optimization budget. Separate reseeding and matching instrumentation should precede further native migration. The present timing decomposition also cannot determine whether tail inflation comes from detector-refresh bursts, feature reseeding, operating-system scheduling, or other variation.

The initial adaptive logs contain one startup/no-track trigger followed by 483 confidence triggers on MOT17-02 and 735 on MOT17-04. Neither run records a displacement or maximum-interval trigger. Confidence is the active limiting condition under the initial parameters. Larger $K$ and $\tau_d$ may leave inference schedules unchanged while confidence remains dominant; lowering $\tau_c$ changes that constraint, but can permit more drift. The completed grid must report these dependencies rather than treating all three parameters as equally influential.

### 5.3. Grid-Search Results Template

The ongoing search is not a completed experiment. Its final selected configurations, uncertainty estimates, and held-out results remain blank in this draft.

| Sequence / split | Policy | $K$ | $\tau_c$ | $\tau_d$ | Reduction (%) | HOTA | MOTA | IDF1 | Mean / p95 latency (ms) | FPS | Repetitions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Development | Baseline | 1 | — | — | 0.00 | [measured] | [measured] | [measured] | [mean / p95] | [mean ± SD] | [n] |
| Development | Selected adaptive | [K] | [threshold] | [threshold] | [measured] | [measured] | [measured] | [measured] | [mean / p95] | [mean ± SD] | [n] |
| Held-out | Same selected policy | [fixed] | [fixed] | [fixed] | [measured] | [measured] | [measured] | [measured] | [mean / p95] | [mean ± SD] | [n] |
| Development | Fixed-K control | [K] | — | — | [measured] | [measured] | [measured] | [measured] | [mean / p95] | [mean ± SD] | [n] |

**Table 6.** Reporting template for completed accuracy–efficiency comparisons. Specify tracker/backend in every policy label. Report corpus-level metrics through evaluator aggregation rather than averaging sequence percentages. A timing SD is unavailable for a single repetition; use an explicit missing-value marker instead of zero.

<div style="background:#F2EDE4;color:#2C2A29;border:1px solid #D9A05B;padding:28px 16px;text-align:center;">
<strong style="color:#A34828;">Figure 1 placeholder — Accuracy–inference Pareto plot</strong><br/>
Insert <code>reports/ablations/pareto_frontier.png</code> after the search is complete. The final export should use charcoal text, terracotta frontier accents, ochre baseline emphasis, and warm-paper backgrounds.
</div>

**Figure 1.** HOTA versus detector-inference reduction for the completed 48-configuration search. Scatter points represent measured configurations; the line connects non-dominated points for the two plotted objectives. The full-inference baseline is shown at 0% reduction. Exported three-objective candidates are separately marked because latency can make a configuration Pareto-optimal without making it non-dominated in this projection. Connecting segments do not imply measurements of intermediate parameter settings. Include the sequence, repetition count, fidelity tolerance, and completion status in the final caption.

### 5.4. Native Microbenchmark Template

The standalone benchmark specifies 10,000 calls per backend per suite. Scheduling comparisons include both native metric packing and a prepacked diagnostic; the primary comparison includes packing. Motion comparisons include conversion of native results to the Python-facing output contract. Warmup, seeded backend order, latency quantiles, process CPU time, sampled RSS, and per-call records are available. Measurements should run after competing experiments finish.

| Suite | Backend | Calls per repetition | Mean (µs) | p50 (µs) | p95 (µs) | p99 (µs) | Mean CPU (%) | Peak RSS (MiB) | Python/native ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scheduling | Python | 10,000 planned | Pending | Pending | Pending | Pending | Pending | Pending | — |
| Scheduling | Native, packing included | 10,000 planned | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Scheduling | Native, prepacked diagnostic | 10,000 planned | Pending | Pending | Pending | Pending | Pending | Pending | Diagnostic only |
| Motion reductions | NumPy reference | 10,000 planned | Pending | Pending | Pending | Pending | Pending | Pending | — |
| Motion reductions | Native, result conversion included | 10,000 planned | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

**Table 7.** Native timing template. Populate from complete uncontended benchmark exports, not the tiny correctness-test fixtures. The ratio is Python mean latency divided by native mean latency: values above 1 favor native execution, while values below 1 favor Python. A local call-rate improvement is not video throughput.

<div style="background:#F9F7F5;color:#2C2A29;border:1px solid #D9A05B;padding:28px 16px;text-align:center;">
<strong style="color:#8C402F;">Figure 2 placeholder — Native latency distributions</strong><br/>
Insert the planned <code>reports/ablations/native_gating_latency.png</code> from the complete per-call microbenchmark records. Use terracotta and ochre for backend distinctions and charcoal labels.
</div>

**Figure 2.** Planned latency distributions for Python and native scheduling/motion reductions, reported in microseconds. Separate native packing-inclusive results from prepacked diagnostics. Show median and p95/p99, the call and repetition counts, feature/track counts, and GIL mode. An ECDF or a shared-bin histogram should expose tails; any logarithmic axis must be labeled. Final values and distributions are pending.

### 5.5. Interpretation and Remaining Experiments

The initial runs show that detector-call reduction and end-to-end throughput do not improve in equal proportion. The pipeline still pays for propagation on skipped frames, and detector updates include expensive matching/reseeding. Native optimization must be assessed against those complete costs.

Before asserting the 50–60% objective, complete the grid, evaluate the initial adaptive MOT17-04 predictions, select a policy under predeclared fidelity limits, and freeze that policy for held-out sequences. Follow with repeated timing trials and a real fixed-K comparison using the same tracker. Evaluate native scheduling and native motion reductions as separate ablations so their effects are not conflated with policy changes.

Occlusion, low texture, low light, scale change, and new-object entry remain stress-test priorities. For power claims, measure synchronized watts or joules on the target device; process CPU/RSS cannot substitute for an energy measurement. Native microbenchmarks must include boundary costs, and real-video tests must establish detector-schedule and tracking parity before a throughput difference is attributed to C++ execution.

## 6. Limitations and Research Scope

The current gate is heuristic and frame-global. One failing object can eliminate savings for all objects in the frame. Consistent flow can drift onto an incorrect surface, while a low-texture correct object can force frequent detection. Translation-only propagation cannot resolve all geometric changes, and IoU association lacks appearance-based identity recovery.

The initial evidence comprises two training sequences and one timing repetition per mode. Background load and thermal conditions were not fully controlled. The absolute accuracy scores remain modest, adaptive MOT17-04 fidelity is pending, and the grid has not completed. Apple Silicon CPU measurements do not establish performance on another edge device. Native correctness, GIL release, and zero-copy input access do not by themselves establish speedup or energy savings.

The final report should retain these boundaries unless new measurements directly address them. The present draft supports a measurable scheduling prototype and a testable native architecture; the stronger claim of 50–60% inference reduction without tracking degradation remains open.

## References

1. Lucas, B. D., and Kanade, T. *An Iterative Image Registration Technique with an Application to Stereo Vision.* IJCAI, 1981. [Original paper](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf).
2. Bewley, A., Ge, Z., Ott, L., Ramos, F., and Upcroft, B. *Simple Online and Realtime Tracking.* ICIP, 2016. [Paper](https://arxiv.org/abs/1602.00763).
3. Dendorfer, P., et al. *MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking.* [Benchmark paper](https://arxiv.org/abs/2010.07548); [official MOT17 sequence archive](https://motchallenge.net/data/MOT17/).
4. Luiten, J., et al. *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking.* arXiv:2009.07736, 2020; published in International Journal of Computer Vision. [Paper](https://arxiv.org/abs/2009.07736).
5. Ultralytics. *YOLO11 model documentation.* [Model and export interface](https://docs.ultralytics.com/models/yolo11/).
6. OpenCV. *Optical Flow.* [Pyramidal Lucas–Kanade documentation](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html).
7. TrackEval. *Multi-object tracking evaluation implementation.* [Source repository](https://github.com/JonathonLuiten/TrackEval). Evaluated package version: 1.3.0.
8. pybind11. *NumPy Interface and Global Interpreter Lock.* [Buffer contracts](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html); [GIL guidance](https://pybind11.readthedocs.io/en/stable/advanced/misc.html).

## Appendix A. Artifact Provenance and Reproduction

The initial results are recorded in the following complete run summaries and their associated prediction/frame logs:

- [MOT17-02 baseline](../results/baseline-real/summary.json) and [adaptive](../results/adaptive-real/summary.json).
- [MOT17-04 baseline](../results/baseline-04/summary.json) and [adaptive](../results/adaptive-04/summary.json).
- [Portable MOT17-02 result record](../docs/results-mot17-02.json).

The saved model SHA-256 is `3586b83d3d87a0465a7df8a85b9c0fb7c8fbac4efacd1dbb5e1ae535707dbc4b`. Initial runtime versions are Python 3.13.3, NumPy 2.5.3, OpenCV 4.14.0.94, ONNX Runtime 1.30.0, SciPy 1.18.1, and psutil 7.2.2. The evaluator uses its separate recorded environment. Version/hash records identify these observations; a newly exported model or changed runtime may produce different detections and timing.

[Reproduction instructions](../docs/reproducibility.md), [grid-search protocol](../docs/grid-search.md), [native API/build notes](../csrc/README.md), [native comparison/integration protocol](../docs/native-benchmark.md), and [plot generation](../docs/pareto-plot.md) document the executable paths. Native validation records distinguish numerical/interface checks from performance experiments. Original datasets and weights remain local and are not redistributed in the repository.

After search completion and with no competing timing workload, the final reporting artifacts can be produced from the repository root:

```bash
.venv-eval/bin/python -m scripts.evaluate_tracking --sequence data/raw/MOT17/train/MOT17-04-FRCNN --run-dir results/adaptive-04
.venv-eval/bin/python -m scripts.plot_pareto
.venv/bin/python -m scripts.benchmark_native --iterations 10000
```

These commands are reproduction/completion instructions and were not executed during preparation of this draft. Figure 2 requires a subsequent distribution-plot export from the benchmark's `calls.jsonl`. Complete results and provenance should replace the placeholders before submission or public presentation.
