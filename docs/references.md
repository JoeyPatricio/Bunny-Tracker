# References

Back to the [project overview](../README.md).

The behaviour vocabulary this project is moving toward is defined in
[ethogram](ethogram.md). Everything below is either a source for
those definitions, directly comparable prior work, or a tool the project uses
or should be measured against.

## Ethogram and behavioural definitions

- NC3Rs / IAT / RSPCA. **Laboratory rabbit ethogram**, from *Refining rabbit
  care: A resource for those working with rabbits in research*, UFAW/RSPCA
  Rabbit Behaviour and Welfare Group (2008).
  https://nc3rs.org.uk/sites/default/files/2022-01/Laboratory%20rabbit%20ethogram.pdf
  Primary source for the v2 state and event definitions. It is itself a
  compilation of the three papers below.
- Morton DB et al. (2003). Refinements in rabbit husbandry: Second report of
  the BVAAWF/FRAME/RSPCA/UFAW joint working group on refinement.
  *Laboratory Animals* 27: 301-329 (Appendix 1, pp. 325-7).
- Held SDE, Turner RJ, Wootton RJ (2001). The behavioural repertoire of
  non-breeding group-housed female laboratory rabbits (*Oryctolagus
  cuniculus*). *Animal Welfare* 10(4): 437-443.
- Gunn D, Morton DB (1995). Inventory of the behaviour of New Zealand White
  rabbits in laboratory cages. *Applied Animal Behaviour Science* 45(3-4):
  277-292. doi:10.1016/0168-1591(95)00627-5

## Prior work on automated rabbit behaviour classification

- Adedeji OJ, Abayomi-Alli A, Arogundade O'T, Abayomi-Alli O, Omoyiola BO
  (2023). **Deep Transfer Learning for Classification of Rabbit Behaviour
  Using Publicly Available Datasets.** *First International Conference on the
  Advancements of Artificial Intelligence in African Context (AAIAC)*, IEEE.
  doi:10.1109/AAIAC60008.2023.10465471

  The closest published work to this project and the natural baseline to
  compare against. YOLOv8 + ByteTrack for rabbit detection and tracking
  (mAP50 0.96), and a fine-tuned ViT-B/16 for behaviour classification over
  seven classes (digging, humping and mating, eating, periscoping, jumping and
  binkying, laying, flopping and loafing), reaching 86% accuracy against 80%
  for EfficientNet and 76% for ResNet18, on roughly 1100 images.

  Differences that matter for this project: their classifier operates on
  **single still images** scraped from YouTube, Roboflow, GitHub and Google
  Images, not on video, so temporal behaviours are inferred from one frame.
  The split is 70:30 train/validation with no test set and, as far as the paper
  describes it, no grouping by source video, so frames from the same clip can
  appear on both sides. Their class list is vernacular and carries no
  operational definitions or inter-observer reliability statistics, and the
  paper reports confusion between the visually similar classes (laying and
  periscoping). The paper is candid that its dataset is "not robust enough to
  cover the numerous ranges of behavioural variations" and that comparative
  benchmarks in this domain barely exist.

## Computational ethology tools

Established tooling in this field. Cited here because any paper coming out of
this project will be expected to position itself against them.

- Friard O, Gamba M (2016). BORIS: a free, versatile open-source event-logging
  software for video/audio coding and live observations. *Methods in Ecology
  and Evolution* 7(11): 1325-1330. https://www.boris.unito.it/
  The standard tool for interval-coded ethogram annotation.
- Bohnslav JP et al. (2021). DeepEthogram, a machine learning pipeline for
  supervised behavior classification from raw pixels. *eLife* 10:e63377.
  doi:10.7554/eLife.63377
  Closest prior art methodologically: supervised behaviour classification
  directly from video, which is what the classifier here does.
- Mathis A et al. (2018). DeepLabCut: markerless pose estimation of
  user-defined body parts with deep learning. *Nature Neuroscience* 21:
  1281-1289.
- Pereira TD et al. (2022). SLEAP: A deep learning system for multi-animal
  pose tracking. *Nature Methods* 19: 486-495.
- Goodwin NL, Nilsson SRO et al. (2024). Simple Behavioral Analysis (SimBA) as
  a platform for explainable machine learning in behavioral neuroscience.
  *Nature Neuroscience* 27: 1411-1424. doi:10.1038/s41593-024-01649-9
- Hsu AI, Yttri EA (2021). B-SOiD, an open-source unsupervised algorithm for
  identification and fast prediction of behaviors. *Nature Communications*
  12: 5188. doi:10.1038/s41467-021-25420-x

## Models and runtime used here

- Qin D et al. (2024). MobileNetV4: Universal Models for the Mobile Ecosystem.
  *European Conference on Computer Vision (ECCV)*. The deployed backbone is
  `mobilenetv4_conv_small`.
- Wightman R (2019). **PyTorch Image Models** (`timm`).
  https://github.com/huggingface/pytorch-image-models
  Source of the frozen backbone and its pretrained weights.
- **ONNX Runtime.** https://onnxruntime.ai/ Inference for the exported
  backbone and temporal head, including the int8 quantized backbone.
- **FFmpeg.** https://ffmpeg.org/ All video capture, segmenting, and frame
  decode, via `imageio-ffmpeg`.
