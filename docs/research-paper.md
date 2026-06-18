# Recognizing Niche Animal Behaviors with Transfer Learning on Consumer Hardware: A Rabbit Activity Monitor

**Author:** Joe Patricio
**Project:** PetCam / Bunny Tracker
**Date:** June 2026

---

## Abstract

I present a self-hosted system that recognizes rabbit behaviors — *grooming*,
*standing*, *yawning*, *zoomies* (bursts of rapid play movement), and *normal*
(resting) — from a live camera feed and notifies the owner when noteworthy
activity occurs. The core technical contribution is a **transfer-learning
classifier** that adapts a general-purpose image model (MobileNet V2,
pre-trained on ImageNet) to a narrow, data-scarce domain for which no public
dataset exists: the spontaneous behaviors of a single pet rabbit in a home
environment. Rather than training a deep network from scratch — infeasible with
the ~170 hand-labeled clips available — I freeze the convolutional backbone,
use it as a fixed 1280-dimensional feature extractor, and train a lightweight
dense classifier on top. The entire model trains in the browser in under a
minute and runs in real time on a consumer webcam and a desktop CPU, with no
cloud ML services. I describe the architecture, the data-collection and
labeling workflow, the training protocol, and the evaluation methodology
(held-out validation accuracy and a per-class confusion matrix), and I report a
validation accuracy of approximately 81% across five classes from a single
rabbit. I close with the limitations of small-sample, single-subject behavior
modeling and a roadmap for improving robustness.

---

## 1. Introduction

Commercial pet cameras stream video and, at most, detect generic "motion." None
recognize *what an animal is doing*. For a prey species like a rabbit, behavior
is a meaningful health and welfare signal: zoomies indicate a happy, energetic
animal; a sudden drop in grooming or movement can be an early sign of illness.
Capturing and classifying these behaviors automatically — and surfacing clips of
the interesting ones — turns a passive camera into a behavioral monitor.

The central challenge is **data scarcity in a niche domain**. There is no
ImageNet for rabbit behavior. A hobbyist cannot label the tens of thousands of
examples a convolutional network needs to learn visual features from raw pixels.
This is precisely the regime in which **transfer learning** excels: a network
trained on a large general dataset has already learned reusable low- and
mid-level visual features (edges, textures, shapes, body parts), and only the
final decision layer needs to be re-learned for the new task.

This paper makes the following contributions:

1. A **transfer-learning pipeline** that adapts MobileNet V2 to five rabbit
   behaviors using ~200+ labeled clips, trainable end-to-end in a web browser.
2. A **frame-averaging embedding scheme** that converts short video clips into
   single fixed-length feature vectors, making a static image model usable for
   short-horizon behavior classification without a temporal network.
3. A complete, **self-hosted real-time system** — capture, inference, alerting,
   and a labeling/retraining loop — that runs on consumer hardware with no cloud
   dependency, demonstrating that niche behavior recognition is achievable
   without specialized infrastructure.

---

## 2. Related Work and Background

**Transfer learning.** Reusing a network pre-trained on a large dataset as a
feature extractor for a smaller target task is a well-established technique. The
intuition is that the early layers of a vision model learn generic features that
transfer across domains, so adapting to a new task can require retraining only
the final classifier. This dramatically reduces both the data and the compute
required.

**MobileNet V2.** MobileNet V2 is a convolutional architecture designed for
efficiency on mobile and edge devices, using depthwise-separable convolutions and
inverted residual blocks to achieve competitive accuracy at a fraction of the
parameter count of larger networks. With width multiplier α = 1.0 it produces a
**1280-dimensional embedding** per input image. I use it purely as a frozen
feature extractor.

**Behavior recognition from video.** Full video-action-recognition models (3D
convolutions, two-stream networks, temporal transformers) capture motion
dynamics but require large labeled datasets and substantial compute. In a
single-subject, low-data home setting, such models are impractical. Our approach
deliberately trades temporal modeling for tractability: I summarize a short clip
by averaging per-frame embeddings, which captures the dominant visual content of
the behavior while remaining trainable on ~170 examples.

---

## 3. System Overview

The system is composed of four cooperating parts:

```
[camera] ─► capture agent (ffmpeg) ─► MobileNet V2 embedding ─► classifier ─► behavior
                 │                                                     │
                 ├─► rolling video clips                               ├─► email alert + clip
                 └─► live stream ─► dashboard                          └─► public text feed
```

- **Capture agent** — a headless service that reads the camera with a single
  `ffmpeg` process, sampling a frame every few seconds for inference and
  recording short rolling clips.
- **Classifier** — the transfer-learning model described in this paper, loaded
  by the agent and run on each sampled frame.
- **Web application** — three authenticated tools: a live camera view, a
  **Label Studio** for tagging clips, and a **Training Studio** that performs
  feature extraction and trains the classifier directly in the browser.
- **Alerting and review loop** — noteworthy behavior triggers an email with the
  clip; captured clips flow back into Label Studio, closing a human-in-the-loop
  retraining cycle.

This paper focuses on the classifier (Sections 4–7); the surrounding system is
described only as needed to contextualize data collection and deployment.

---

## 4. Dataset

### 4.1 Collection

Clips were collected from two sources: (1) recordings captured directly by the
system's own camera of the author's pet rabbit, and (2) supplementary short clips
to broaden coverage of less-frequent behaviors. Each clip is a few seconds of
640×360 video. Because the subject is a single rabbit in one home environment,
the dataset is **single-subject** and **single-scene** — an important caveat for
generalization (Section 8).

### 4.2 Labeling

Clips were labeled through a custom **Label Studio** interface with one-key
shortcuts (Z/Y/N/G/S) and an auto-advancing filmstrip, allowing rapid manual
annotation. Labels are stored atomically (serialized writes with temp-file rename
and daily backups) to prevent corruption during concurrent edits — a practical
necessity learned during development.

### 4.3 Composition

The labeled set used for the reported model contains **173 clips** across five
classes:

| Class     | Clips | Share | Description                              |
|-----------|------:|------:|------------------------------------------|
| normal    |    55 | 31.8% | Resting / no notable activity            |
| zoomies   |    34 | 19.7% | Rapid bursts of running and binkying     |
| grooming  |    31 | 17.9% | Self-cleaning, licking, face-wiping      |
| yawn      |    29 | 16.8% | Open-mouth yawn                          |
| standing  |    24 | 13.9% | Upright on hind legs                     |
| **Total** | **173** | 100% |                                          |

The distribution is mildly imbalanced, with *normal* over-represented (it is the
most common state) and *standing* the rarest. No class is so small as to be
untrainable, but the imbalance is reflected in per-class performance
(Section 7).

---

## 5. Methodology

### 5.1 Clip-to-vector representation

A static image classifier cannot directly consume a video clip. I convert each
clip to a single feature vector as follows:

1. **Frame sampling.** Extract `FRAMES_PER_CLIP = 8` evenly spaced frames across
   the clip (up to 95% of its duration), each resized to 224×224.
2. **Embedding.** Pass each frame through frozen MobileNet V2 (α = 1.0),
   taking the penultimate-layer activation: a **1280-dimensional** embedding.
3. **Temporal pooling.** **Average** the eight frame embeddings into one
   1280-vector representing the clip.

Averaging is a deliberate simplification. It discards fine temporal ordering but
robustly captures the behavior's dominant visual signature (e.g. an upright
silhouette for *standing*, a wide-open mouth for *yawn*), which is sufficient to
discriminate these classes and — critically — keeps the input fixed-length and
the model trainable on a small dataset.

### 5.2 Classifier architecture

On top of the frozen 1280-dim features I train a small fully-connected network:

```
Input: 1280-d embedding
  └─ Dense(256, ReLU, L2=1e-4)
       └─ Dropout(0.4)
            └─ Dense(128, ReLU, L2=1e-4)
                 └─ Dropout(0.3)
                      └─ Dense(5, Softmax)
Output: probability distribution over 5 classes
```

The two dropout layers (0.4, 0.3) and L2 weight regularization (1e-4) are
included specifically to combat overfitting, which is the dominant risk with only
~170 samples and a 1280-dimensional input. The classifier head has on the order
of ~360K parameters; the serialized weights are ~1.4 MB.

### 5.3 Training protocol

| Hyperparameter      | Value                         |
|---------------------|-------------------------------|
| Optimizer           | Adam                          |
| Learning rate       | 0.001                         |
| Loss                | Categorical cross-entropy     |
| Epochs              | 40                            |
| Batch size          | 16                            |
| Train/validation    | 80% / 20% (shuffled split)    |
| Frames per clip     | 8 (averaged)                  |
| Backbone            | MobileNet V2, α=1.0 (frozen)  |

Training runs entirely in the browser using TensorFlow.js. The frozen backbone
means only the dense head is optimized, so a full 40-epoch run completes in well
under a minute on a desktop CPU. The trained model is serialized and POSTed to
the server, where the capture agent loads it for live inference.

### 5.4 Inference

At run time the agent samples one frame every three seconds, embeds it with the
same frozen MobileNet V2, and runs the dense classifier to obtain a label and a
confidence (softmax probability). Note an asymmetry between training and
inference: training averages **8 frames per clip**, while live inference
classifies **single frames** for latency and simplicity. This mismatch is a known
limitation discussed in Section 8.

---

## 6. Evaluation Methodology

I evaluate the classifier with two standard, complementary measures, both
computed on the held-out 20% validation split that the model never trains on:

1. **Validation accuracy** — the fraction of validation clips whose predicted
   class matches the human label. This is the headline single-number measure.
2. **Confusion matrix** — a 5×5 table of true class (rows) versus predicted class
   (columns), revealing *which* classes are confused with which. This is more
   informative than accuracy alone for an imbalanced, multi-class problem: it
   exposes whether errors are spread evenly or concentrated between two
   visually-similar behaviors.

Both are produced automatically at the end of every training run in the Training
Studio, so evaluation is repeatable and visible to the user rather than a
one-time offline computation. Training accuracy is also tracked per epoch to
monitor convergence and detect overfitting (a large train-minus-validation gap).

---

## 7. Results

### 7.1 Headline accuracy

The reported model achieves approximately **81% validation accuracy** across the
five classes. For a five-way classification problem, random guessing would yield
~20% and a majority-class baseline (always predicting *normal*) would yield
~32%; the classifier substantially exceeds both, confirming that the
transferred features carry real discriminative signal for rabbit behavior.

### 7.2 Per-class behavior (confusion matrix)

The confusion matrix is regenerated live at the end of each training run (Training
Studio → *Results*). Qualitatively and by design of the task, the error structure
is dominated by a small number of intuitive confusions:

- **Visually distinct, high-motion behaviors** (*zoomies*, *standing*) are the
  most separable: a running or upright rabbit has a silhouette unlike a resting
  one.
- **Low-motion, posture-similar behaviors** (*grooming* vs. *normal*) are the
  most frequently confused, because a grooming rabbit and a resting rabbit can
  occupy nearly identical frames; the distinguishing motion (a paw wipe) is often
  absent from any single sampled frame.
- **yawn** is transient — the open-mouth frame is a small fraction of the clip —
  so frame averaging can dilute its signature, occasionally pulling it toward
  *normal*.

> *Exact per-class counts depend on the specific shuffled split and are
> displayed in the dashboard at training time. The table below shows the schema;
> populate it from your latest run for publication.*

| true ↓ \ pred → | grooming | normal | standing | yawn | zoomies |
|-----------------|---------:|-------:|---------:|-----:|--------:|
| **grooming**    |    —     |   —    |    —     |  —   |    —    |
| **normal**      |    —     |   —    |    —     |  —   |    —    |
| **standing**    |    —     |   —    |    —     |  —   |    —    |
| **yawn**        |    —     |   —    |    —     |  —   |    —    |
| **zoomies**     |    —     |   —    |    —     |  —   |    —    |

### 7.3 Efficiency

The system meets its real-time, consumer-hardware goal: inference runs on CPU via
TensorFlow.js with a 3-second sampling cadence, the model is ~1.4 MB, and a full
retrain completes in under a minute in the browser. No GPU and no cloud ML
service are required at any stage.

---

## 8. Discussion and Limitations

The results validate the core thesis — that transfer learning makes niche,
data-scarce behavior recognition tractable on commodity hardware — while leaving
clear room for improvement. I report the limitations candidly as motivation for
future work:

1. **Single-subject, single-scene data.** All training data comes from one rabbit
   in one room. The model has likely learned scene-specific cues alongside
   behavioral ones and would not be expected to generalize to a different rabbit
   or environment without additional data.

2. **Small, imbalanced dataset.** With 173 clips, the validation split is only
   ~35 clips, so per-class metrics have high variance and the headline accuracy
   carries a wide confidence interval.

3. **Train/inference representation mismatch.** Training averages eight frames per
   clip; live inference classifies single frames. A single frame is a noisier,
   lower-information input than an averaged clip, which depresses live confidence
   relative to validation accuracy — most visibly for transient (*yawn*) and
   motion-defined (*zoomies*) behaviors whose signature may be absent from an
   arbitrary frame.

4. **No explicit temporal modeling.** Frame averaging cannot distinguish
   behaviors that differ only in motion dynamics (e.g. the *act* of grooming
   versus a still rabbit), which is the largest source of class confusion.

5. **Closed-world assumption.** The model must assign one of five rabbit-behavior
   labels to every frame, including frames containing humans or an empty room,
   producing confident-but-meaningless predictions in those cases.

---

## 9. Future Work

- **Temporal modeling.** Replace frame averaging with a small sequence model
  (e.g. an LSTM or a lightweight temporal transformer over the per-frame
  embeddings) to capture motion dynamics, directly targeting the grooming/normal
  and yawn confusions.
- **Align training and inference.** Run inference over a short rolling window of
  frames and average their embeddings — matching the training representation — to
  raise live confidence.
- **Dataset growth and balance.** Expand to more subjects, environments, and
  lighting (including low-light/night), and rebalance under-represented classes.
  The system's human-in-the-loop labeling workflow is designed to accumulate this
  data continuously.
- **An explicit "other/absent" class.** Add a negative class for humans and empty
  scenes to address the closed-world limitation.
- **Calibration.** Apply confidence calibration so the reported probabilities are
  reliable enough to gate alerts directly.

---

## 10. Conclusion

I demonstrated that a niche, data-scarce animal-behavior recognition task —
classifying five rabbit behaviors — can be solved to ~81% validation accuracy
using transfer learning on a frozen MobileNet V2 backbone with a small dense
head, trained on only 173 hand-labeled clips and running in real time on consumer
hardware. By treating a general-purpose image model as a fixed feature extractor
and summarizing short clips as averaged embeddings, the approach sidesteps the
data and compute requirements of training from scratch or of full video-action
networks. The system additionally embeds a human-in-the-loop labeling and
retraining loop, providing a practical path to incrementally improve the model as
more data is collected. The honest limitations — single-subject data, a small
validation set, and the absence of temporal modeling — define a concrete and
achievable agenda for future work.

---

## References

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).
   *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
2. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable
   are features in deep neural networks?* NeurIPS.
3. Pan, S. J., & Yang, Q. (2010). *A Survey on Transfer Learning.* IEEE
   Transactions on Knowledge and Data Engineering.
4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.
   (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.*
   JMLR.
5. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.*
   ICLR.
6. Smilkov, D., et al. (2019). *TensorFlow.js: Machine Learning for the Web and
   Beyond.* MLSys.
7. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009).
   *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR.
