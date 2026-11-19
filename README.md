# CRC-LN-Assessment

**Colorectal lymph node assessment**

PyTorch implementation of patient-level CT and optical learning. The diagnostic network combines phase-aware CT aggregation, optical multiple-instance pooling and cross-modal residual fusion. Available branches provide predictions when an input modality is missing.

## Installation

```bash
pip install -e .
```

## Training

Provide stage-specific input manifests with explicit patient roles (`development`, `test`). `Dataset` in `src/crc_ln/data.py` defines the input adapter for CT candidate tensors and optical ROI images. Development records provide binary patient labels. Test assignments remain separate. File paths and input hashes bind the prepared inputs to each run.

```bash
crc-ln-train --config configs/stage1.json --manifest stage1_inputs.json --imagenet resnet34.pth --output runs/stage1 --device cuda:0
crc-ln-train --config configs/stage2.json --manifest stage2_inputs.json --initial runs/stage1/best.pt --output runs/stage2 --device cuda:0
```

Stage 1 learns CT representations with random arterial-phase masking when both CT branches are available. Stage 2 optimizes the multimodal network. The configuration controls the update limit, monitoring and checkpoint intervals, and early-stopping patience. The split seed fixes the development partition independently of model initialization. Validation loss determines the retained checkpoint, and its validation predictions determine output-specific Youden thresholds. Recovery checkpoints include optimizer, sampler and random-generator states.

## Inference

```bash
crc-ln-predict --checkpoint runs/stage2/best.pt --manifest inference_inputs.json --thresholds runs/stage2/thresholds.json --output predictions.json --device cuda:0
```

The inference adapter accepts prepared candidate inputs. The output contains branch probabilities, the available-input route and fixed-threshold predictions.

## Layout

- `src/crc_ln/`: model, input adapter, training and inference.
- `configs/`: two-stage optimization settings.
