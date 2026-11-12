# CORVUS — Neural Network and Language Model from Scratch in C++17

No heavy dependencies: only C++17 + `stb_image.h` (included in the project), with optional OpenMP.

Features: tabular data, image processing (CNN), text classification, and a **GPT language model** capable of chatting and generating text.

## Building

**With CMake (recommended, especially for IDEs):** Open the `minnn` directory (the one containing `CMakeLists.txt).

```bash
cmake -S . -B build            # Release is the default
cmake --build build -j
ctest --test-dir build         # Same as minnn selftest
./build/minnn selftest
```

- Use `-DCMAKE_BUILD_TYPE=Debug` for debugging; use `-DMINNN_NATIVE=OFF` if you plan to run the executable on a different CPU.
- If `include/third_party/stb_image.h` is missing, CMake downloads it automatically (`-DMINNN_FETCH_STB=OFF` disables this).
- If any project files are missing, CMake reports the names of all missing files. **Copy the entire directory** (or use `minnn.zip`), not individual files.
- Tested with GCC 13 and 14, and Clang, in both Release and Debug modes. MSVC has not been tested.

**Without CMake:**

```bash
g++ -std=c++17 -O3 -march=native -fopenmp -Iinclude src/main.cpp src/stb_impl.cpp -o minnn
./minnn selftest        # All checks should PASS (currently 56 checks)
```

## Running and Starting Training (Visual Studio)

1. Select `File ← Open ← Folder` and open the `minnn` directory (the one containing `CMakeLists.txt`).
2. At the top, open the configuration menu next to the configuration name (for example, `x64-Debug`) → `Manage Configurations…` → `+` → **`x64-Release`**, then select it. **Debug is about 11 times slower** (measured: 1,064 vs. 11,879 tokens/second). Always use Release for training.
3. Select `Build ← Build All`. The executable is located at `out/build/x64-Release/minnn.exe` (the exact path is shown in the Output window).
4. Running the program without arguments prints the help text. There are two ways to pass arguments:
   - **Terminal (easiest):** Select `View ← Terminal`, then run a command such as `out\build\x64-Release\minnn.exe train-lm --text data/corpus --preset small --out data/lm.bin` from the project directory.
   - **Inside the IDE:** Use `Debug ← Debug and Launch Settings for minnn` to create `launch.vs.json`, then set `"args"` and `"currentDir": "${workspaceRoot}"` (see `tools/launch.vs.json.example` for an example). Press `Ctrl+F5` to run without the debugger and keep the console open.
- Quick checks: `minnn.exe selftest` (should report 47/47) and `minnn.exe demo`.
- The Windows console is configured for UTF-8, but the legacy console may display Persian characters as disconnected or reversed (a display-only issue; the output itself is correct). **Windows Terminal** is recommended.
- Tested: the Windows version was compiled with MinGW (build only, not executed). It has not been tested inside Visual Studio/MSVC.

## CSV Files and SQL Data

**CSV — what each command expects** (columns can be specified either by index or by **column name** in the header):

| Command | Required columns | Options |
|---|---|---|
| `train` (tabular) | Numeric columns + one label column | `--label-col` (defaults to the last column), `--task classification\|regression` |
| `train-text` | One text column + one label column | `--text-col`, `--label-col`, `--header 0` if there is no header |
| `train-lm --chat` | Question + answer | `--prompt-col`, `--response-col`, `--header 0` |
| `train-lm --text f.csv` | One text column only | `--text-col comment` |

- The column delimiter is **detected automatically**: `,`, `;` (used by Excel under some regional settings), or Tab. Other delimiters (such as `|`) are not detected; the program reports a clear error rather than silently training on corrupted data.
- UTF-8, BOM, and Windows CRLF are supported. Decimal numbers must use a period (`1.5`, not `1,5`); the program displays a helpful message if it encounters decimal commas.
- **Empty cells:** An empty numeric cell causes an error that includes the row and column numbers. An empty label cell produces a **warning** (rows with empty labels become a separate class). It is best to filter these rows out beforehand.

**SQL — minnn does not read SQL files directly; convert them to CSV first** using `tools/sql_to_csv.py` (Python 3.8+, no packages to install):

```bash
python3 tools/sql_to_csv.py --sql dump.sql --list                      # List tables, columns, and row counts
python3 tools/sql_to_csv.py --sql dump.sql --table comments --out data/comments.csv
python3 tools/sql_to_csv.py --db  site.sqlite --query "SELECT text, label FROM comments WHERE label IS NOT NULL AND label != ''" --out data/c.csv
python3 tools/sql_to_csv.py --sql chat.sql --query "SELECT question AS prompt, answer AS response FROM chat" --out data/chat.csv
```

- Input formats: SQLite database files (`--db`), SQLite application dumps, and **MySQL / MariaDB** dumps (mysqldump, phpMyAdmin) via `--sql`. **PostgreSQL** dumps are not supported (the tool explains that you should export CSV from the database using `psql \copy`).
- For older Persian dumps that are not UTF-8, use `--encoding cp1256`. NULL and BLOB values are converted to empty cells, so filter them in `--query` (for example, `WHERE ... IS NOT NULL`).
- Benchmark: a 48 MB dump (200,000 rows) was converted in 7 seconds and used approximately **6 times the dump size in RAM** (estimated: a 1 GB dump requires about 6 GB of RAM).
- Full workflow: SQL → CSV → `train-text --data data/c.csv --text-col text --label-col label` (it reached 100% accuracy on my sample dataset; real-world data will produce different accuracy).
- Tests: `python3 tools/test_sql_to_csv.py` (11 tests), plus a separate randomized test: 500 rows of hostile text (quotes, backslashes, newlines, emoji, SQL keywords) in a MySQL-format dump, totaling 2,000 cells, with no discrepancies whatsoever.

## Where to Put Training Data

Anywhere you like; specify the path with `--text`, `--chat`, or `--data`. Recommended: a **`data/`** directory alongside the project (full explanation and format table: `data/README.md`).

- Raw text: `.txt` files in `data/corpus/` → `train-lm --text data/corpus`
- Chat data: `data/my_chat.csv` with `prompt,response` columns → `train-lm --chat data/my_chat.csv`
- Relative paths are resolved from the **working directory** (in Visual Studio, the default is the executable's directory, not the project directory). Use `currentDir` or an absolute path. If a file cannot be found, the error message prints the working directory.
- Files must be UTF-8. BOM and Windows CRLF are stripped automatically.

## Project Structure

| File | Purpose |
|---|---|
| `tensor.hpp` | Matrix operations and matrix multiplication (register-blocked, OpenMP) |
| `layers.hpp` | Dense, Activation, Dropout, Conv2D, MaxPool2D, Embedding, BatchNorm2D, GlobalAvgPool, ResidualBlock |
| `loss.hpp` / `optimizer.hpp` | CrossEntropy, MSE / SGD, Adam(W) with state saving |
| `dataset.hpp` `text.hpp` | CSV handling, Persian-friendly tokenizer |
| `imageproc.hpp` | `Image` class (anti-aliased resize, fit), augmentation transforms, `ImageAugmenter` |
| `image.hpp` | Image loading; `ImageSource` classes (`FolderImageSource`, `IdxImageSource`) |
| `model.hpp` | `ModelBuilder` (MLP / CNN / Text), `Model`, saving/loading |
| `trainer.hpp` | `Trainer` + `TrainCallback` (`ProgressLogger`, `EarlyStopping`, `CheckpointSaver`) |
| `command.hpp` | `Command` / `CommandRegistry` (all CLI commands) |
| `bpe.hpp` | Byte-level BPE tokenizer (supports any UTF-8 text) |
| `transformer.hpp` | GPT with manual backward pass, KV cache, `growLM`, `estimateLM` |
| `lm.hpp` | `train-lm` / `chat` / `generate` / `info` / `grow` |
| `selftest.hpp` | Automated tests for all components |

## Language Model (Chat)

Data: a CSV file with `prompt,response` columns and/or raw text (`.txt` files or a directory).

```bash
python3 tools/make_sample_chat.py                       # Generate sample data
./minnn train-lm --chat sample_chat.csv --preset tiny --steps 900 --lr 2e-3 --out lm.bin
./minnn chat --model lm.bin                             # Chat (/reset, /quit)
./minnn generate --model lm.bin --prompt "سلام" --max-new 100
./minnn train-lm --text corpus/ --chat pairs.csv --preset small   # Pretraining + chat training together
```

- Pressing `Ctrl+C` during training saves `lm.bin.last` + `lm.bin.last.opt`; resume with `--resume lm.bin.last`.
- `--resume` restores the weights and optimizer state (`.opt`).
- `--accum K`: effective batch size = batch × K, without increasing RAM usage.
- Sampling options: `--temp 0.8 --top-k 40 --top-p 0.95 --rep-penalty 1.0` (try `1.1` if the model repeats too much).

## “Scaling Up to Large”: What Is Actually Required

Model size is determined by **data and computation**, not by a single switch. This project provides three tools:

1. **`--preset tiny|small|medium|large|xl`** — configures model size, vocabulary, batch size, and learning rate together (any explicitly supplied flag overrides the preset).
2. **`./minnn info --preset large`** — estimates parameters, RAM usage, and computational cost before training starts.
3. **`./minnn grow --model lm.bin --layers 12 --ctx 512`** — enlarges a trained model without losing what it has learned (new layers are identity layers; this has been verified by tests). Then run `train-lm --resume`.
   - Only depth and context length can be increased, not width (`dim`).
   - **Adding layers:** loss does not change (measured: 0.028 → 0.029). **Increasing context length:** new positions have not been trained, so loss temporarily increases until they are trained.

Estimates are produced by `info` itself (RAM and FLOPs). The “8 GFLOPS” column is based on my actually measured speed on one core (~8,900 tokens/second for `tiny`). The “50 GFLOPS” column is **my estimate** for 8 threads and has not been measured — check the `tok/s` line at the beginning of your own training run to see your actual speed.

| Preset | Parameters | Training RAM | Required tokens (~20× parameters) | Time at 8 GFLOPS (measured, 1 core) | Time at 50 GFLOPS (estimated, 8 threads) |
|---|---:|---:|---:|---:|---:|
| `tiny` | 0.14M | 0.02 GB | 3M | 6 minutes | 1 minute |
| `small` | 0.97M | 0.13 GB | 19M | 4.1 hours | 39 minutes |
| `medium` | 5.64M | 0.85 GB | 113M | 6 days | 22.7 hours |
| `large` | 28.69M | 2.4 GB | 574M | 0.4 years | 25 days |
| `xl` | 92.19M | 6.9 GB | 1,844M | 4.5 years | 0.7 years |

**A reality check:**

- Up to `small` is practical on the CPU in my system; `medium` takes tens of hours; `large` and `xl` are effectively impractical on a CPU (and the RX 580 is not used yet; I have not implemented a GPU backend).
- A larger model without enough data performs worse, not better (overfitting). Rule of thumb: tokens ≈ 20× the number of parameters. If your dataset is only a few megabytes, `tiny` or `small` is the right choice.
- A model trained from scratch on personal data learns to imitate patterns in that data. The broad knowledge of large models comes from billions of tokens. For quality comparable to ChatGPT, fine-tune a pretrained open-source model instead of scaling up this project.

## Image Processing

```bash
./minnn train-image --data data/ --size 64 --channels 3 --fit crop --conv 16,32,64 --augment --epochs 40
./minnn predict --model model.bin --data photo.jpg --top 3 --tta     # TTA: average predictions for the original and mirrored image
```

Directory structure: `data/<class>/*.png|jpg|bmp|...` (or MNIST files using `--idx-images/--idx-labels`).

- **Proper resizing:** downscaling uses area averaging (anti-aliasing); upscaling uses bilinear interpolation. Use `--fit stretch|crop|pad` for different aspect ratios.
- **Normalization** is performed per channel and calculated using training data only.
- **Augmentation** is applied to raw pixels (before normalization): `--augment`, or individual options such as `--rotate DEG --zoom 0.9,1.1 --shift 0.1 --hflip --vflip --jitter 0.2 --noise 0.02 --cutout 0.3`.
- **Architecture:** `--arch plain|resnet`, `--bn`, `--gap`, `--blocks N`.

### Measured Results (Actual Measurements, Not Claims)

A small, deliberately challenging benchmark: training on 200 clean images and testing on 320 “dirty” images (rotation up to ±30°, translation, scaling, and noise). Four classes, 60 epochs, batch size 16, average of 3 seeds. Reproduce it with `python3 tools/make_bench_images.py` and `tools/score_predictions.py`.

| Model | Accuracy on the corrupted test set |
|---|---:|
| plain, no augmentation | 76.9% |
| plain + `--augment` | 85.7% |
| plain + `--bn` + `--augment` | 86.1% |
| Two-stage ResNet + `--augment` | 57.2% |

- **Augmentation added about 9 percentage points**, making it the clearest improvement.
- **BatchNorm made no meaningful difference on this small network** (the difference was within the noise; convergence speed was also the same). It has not been tested on deeper networks.
- **The shallow ResNet performed worse across all 3 seeds** (100% training accuracy, 57% test accuracy). It improved with greater depth (`--conv 8,16,32 --blocks 2`: 78.7% on one seed), but still scored below the plain model. I suspect the cause is a small receptive field, but this has not been proven. Conclusion: ResNet is not a free upgrade on small datasets.
- Limitation: this is a small, synthetic benchmark; results may differ on real-world images.

## Text and Tabular Data

```bash
./minnn train-text  --data comments.csv --ngrams 2
./minnn train       --data data.csv --hidden 128,64
./minnn predict     --model model.bin --data ...     # Works with every model type
./minnn info        --model model.bin
```

For details on every option, run `./minnn` without arguments. `--checkpoint` writes the best model to disk whenever an improvement is made.

## Object-Oriented Design

| Concept | Classes | Benefit |
|---|---|---|
| Layers | `Layer` ← `Dense`, `Conv2D`, `BatchNorm2D`, `ResidualBlock` (a Layer that contains other Layers) | Each new layer only needs to implement `forward/backward` |
| Model construction | `ModelBuilder` ← `MLPBuilder`, `CNNBuilder`, `TextBuilder` | A new model type only requires a new builder |
| Image transforms | `ImageTransform` ← `RandomAffine`, `ColorJitter`, `Cutout`, ... and `Compose` | Augmentations can be composed |
| Data sources | `ImageSource` ← `FolderImageSource`, `IdxImageSource` | Add a new source (e.g. CSV paths) without changing the training code |
| Training | `Trainer` + `TrainCallback` ← `ProgressLogger`, `EarlyStopping`, `CheckpointSaver` | New behavior (e.g. logging to a file) only requires a callback; the training loop remains untouched |
| CLI | `Command` ← `TrainImageCommand`, `FnCommand`; `CommandRegistry` | Adding a command requires one registration line |

For transparency: `TrainImageCommand` is a real class with separate methods for each stage. The other commands are currently wrapped around legacy functions using `FnCommand` adapters. Refactoring LM training (`train-lm`) to use `Trainer` has not been done yet; it still has its own training loop.

## Testing

`./minnn selftest` checks the following: matrix multiplication against an exact reference, backpropagation for all layers using finite differences (Conv and Pool are checked separately and precisely), full GPT backpropagation, tokenizer round-tripping, KV-cache output against the training forward pass, saving/loading, `grow`, Adam resumption, CSV parsing and the Persian tokenizer, and whether GPT actually reduces its loss.

The tests have been evaluated using **mutation testing**: intentionally breaking backward-pass implementations (attention, LayerNorm, GELU, Conv, Pool, and the head) must cause tests to fail — all 8 mutations were detected.

**Object-oriented behavior tests:** `Trainer` with stop callbacks, `EarlyStopping` (counter, reset, restoration of the best weights), `CheckpointSaver`, `CommandRegistry`, and `ImageSource` are each tested and evaluated with mutation testing (6 intentional faults; all 6 were detected). After the refactor, training behavior is byte-for-byte identical to the pre-refactor model when using the same seed.
