# Basically U-Net

A U-Net-style convolutional neural network for binary image segmentation, built with PyTorch.

"Be the square among the circles, and they will segment you."
                                             — Unknown author


---

## Project Structure

```
project/
├── model.py          # Model architecture and Dice loss
├── data.py           # Dataset loader
├── main.py           # Training and evaluation script
├── dataset/
│   ├── train/        # Training images and labels
│   └── test/         # Test images and labels
└── weights/          # Saved model checkpoints and logs
```

---

## Dataset Format

Images and labels must be in the **same folder**, named as follows:

```
image_1.png
image_2.png
...
label_1.png
label_2.png
...
```

All images are expected to be **PNGs**. The dataset loader will automatically pair each `image_N.png` with its corresponding `label_N.png` and warn if any label is missing.
The task for this dataset is that the model can segment a square amoung circles.

---

## Model Architecture

This implementation is based on the U-Net architecture from [Ronneberger et al. (2015)](https://arxiv.org/pdf/1505.04597).

`BasicallyUnet` is a 4-level encoder-decoder network with skip connections.


```
Input (1, H, W)
      │
   Encoder
  64 → 128 → 256 → 512
      │
  Bottleneck
      1024
      │
   Decoder
  512 → 256 → 128 → 64
      │
  Final Conv + Sigmoid
      │
Output (1, H, W)  ∈ [0, 1]
```

Each encoder block applies Conv→BN→ReLU, saves a skip connection, then MaxPool. Each decoder block upsamples, concatenates the skip, then applies Conv→BN→ReLU. The final head is a 1×1 convolution followed by Sigmoid, producing a single-channel probability mask.

---

## Installation

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

---

## Usage

### Training

```bash
python main.py \
  -job        TRAIN \
  -data_path  ./dataset/train \
  -test_path  ./dataset/test \
  -model_file ./weights/model.pt \
  -log_file   ./weights/log.csv \
  -num_epoch  100
```

To **resume** from a checkpoint, add:

```bash
  -state_file ./weights/model_50.pt
```

### Testing

```bash
python main.py \
  -job        TEST \
  -test_path  ./dataset/test \
  -model_file ./weights/model.pt
```

Output masks and input images are saved to the test folder as `output{N}_label.png` and `output{N}_image.png`. A `loss.csv` summary is also written there.

---

## Arguments

| Argument | Description |
|---|---|
| `-job` | `TRAIN` or `TEST` (required) |
| `-data_path` | Path to training dataset |
| `-test_path` | Path to test/validation dataset |
| `-model_file` | Path to save/load model weights |
| `-log_file` | Path to save training log CSV |
| `-num_epoch` | Number of training epochs |
| `-state_file` | Checkpoint to resume training from |

---

## Training Configuration

The following hyperparameters can be adjusted at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `num_batch` | 10 | Batch size |
| `num_workers` | 4 | DataLoader worker threads |
| `learning_rate` | 1e-3 | Initial learning rate |
| `min_learning_rate` | 1e-5 | Minimum learning rate (LR floor) |
| `nfilter` | 64 | Base channel count |
| `save_frequency` | 10 | Save checkpoint every N epochs |
| `augment_noise` | 0.025 | Gaussian noise std for augmentation |

The learning rate follows an **exponential decay** schedule (`×0.975` per epoch) with a hard floor at `min_learning_rate`.

---

## Data Augmentation

When `noise` is set, the dataset applies:
- **Random horizontal flip** (50% probability, applied consistently to image and label)
- **Gaussian noise** added to the input image only (std sampled uniformly from `[0, noise]`)

---

## Output

After training, the `weights/` folder will contain:

- `model.pt` — final model weights, optimizer state, and loss history
- `model_00.pt`, `model_10.pt`, ... — periodic checkpoints
- `log.csv` — per-epoch train loss, test loss, and time

---

## License

Copyright © 2025 Etore Maloso Tronconi 