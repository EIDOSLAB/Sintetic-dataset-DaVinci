# %% [markdown]
# ## Import

# %%
# !pip3 install opencv-python-headless
# !pip3 install albumentations
# !pip3 install pytorch_lightning
# !pip3 install segmentation_models_pytorch
# !pip3 install huggingface_hub

# !pip3 install wandb
# !pip3 install torchmetrics




# %%
import os
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import math
import shutil
import subprocess
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
import torch
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import os
import torch
import zipfile
from huggingface_hub import hf_hub_download
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# %% [markdown]
# ## Pre-Processing Configuration

# %%
import sys, os, torch
print("python exe:", sys.executable)
print("torch:", torch.__version__, " wheel CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

if "Tesi-Borra" not in os.getcwd():
    os.chdir("/scratch/Tesi-Borra")
print("dir :", os.getcwd())

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--architecture", type=str, default="resnet18", help="architecture")
parser.add_argument("--dataset", type=str, default="DaVinci", help="dataset")
parser.add_argument("--name", type=str, default="DaVinci", help="name")
parser.add_argument("--epochs", type=int, default=20, help="epochs")
parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
parser.add_argument("--train", type=float, default=0.8, help="train")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu" , help="device hardware")

try:
    conf = parser.parse_args()
except:
    conf = parser.parse_args([])

conf.name = conf.name + f"_alpha={conf.alpha}"

print("Configurazione da dati: " ,conf)
# conf={
#         "learning_rate": 0.02,
#         "architecture": "resnet50",
#         "dataset": "DaVinci",
#         "epochs": 10,
#         "alpha" : 0.5,
#         "train": 0.8,
#         "seed" : 42,
#         "batch_size": 8,
#         "device": "cuda" if torch.cuda.is_available() else "cpu" ,
        
# }

# %% [markdown]
# Wandb LOGIN

# %%

# wandb.login(key="99fb6e2a935e54f90a3613657ba8a627064bf327")
wandb.login(key="8a901b6f2fe21bc78ccc8588e16dc1541da1f7f4")
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="eidos-giorgio",
    # Set the wandb project where this run will be logged.
    project="Tesi-Borra",
    # Track hyperparameters and run metadata.
    config=conf,
    name= conf.name,
)
wandb_logger = WandbLogger(experiment=run)



# %% [markdown]
# ## Download Real Dataset

# %%


def ensure_dataset(local_dir="data", repo_id="rossbina/tesi-dataset", filename="dataset.zip"):
    os.makedirs(local_dir, exist_ok=True)

    # Scarica in cache da HF (se già presente, non riscarica)
    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename
    )

    # Estrai solo se non già estratto
    extract_flag = os.path.join(local_dir, ".extracted.flag")
    if not os.path.exists(extract_flag):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(local_dir)
        with open(extract_flag, "w") as f:
            f.write("ok")

    return local_dir


# %%
data_dir = ensure_dataset(local_dir="/scratch/Tesi-Borra/dataset") 

# %%
import shutil
import os

output_dir = "/scratch/Tesi-Borra/dataset/dataset/DaVinci"+str(conf.alpha)
os.makedirs(output_dir, exist_ok=True)
# elimina tutto il contenuto della cartella
for item in os.listdir(output_dir):
    item_path = os.path.join(output_dir, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

print("Contenuto della cartella DaVinci eliminato!")


# %% [markdown]
# ## Split dataset synt + real

# %%
ALPHA = conf.alpha
TRAIN = conf.train

# %%

# ===================== CONFIGURAZIONE =====================
DATASET_DIR = Path("/scratch/Tesi-Borra/dataset/dataset")   # contiene: img, mask, img_sint, mask_sint
OUTPUT_DIR  = Path("/scratch/Tesi-Borra/dataset/dataset/DaVinci"+str(conf.alpha))   # output finale con: train_img, train_mask, valid_img, valid_mask, test_img, test_mask

USE_SYMLINKS = True  # True: creo symlink, False: copio i file

# Split desiderato sui REALI (per 80/10/10 usa 0.8, 0.1, 0.1)
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = TRAIN, (1-TRAIN)/2, (1-TRAIN)/2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}

# ====== CONFIG PIPELINE SINTETICI ======
UTILS_DIR = Path("/scratch/Tesi-Borra/Utils")
MAYAPY = "/Applications/Autodesk/maya2024/Maya.app/Contents/bin/mayapy"

# Numero minimo di run per iterazione (anche se il deficit è piccolo)
MIN_RUNS_PER_ITER = 3
# Numero massimo di iterazioni della pipeline per sicurezza
MAX_GENERATION_ITERS = 10

# ===================== UTILITY =====================

def list_files_by_stem(root: Path) -> Dict[str, Path]:
    if not root.exists():
        return {}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: str(p.relative_to(root)).lower())
    by_stem: Dict[str, Path] = {}
    for p in files:
        stem = p.stem
        if stem not in by_stem:
            by_stem[stem] = p
    return by_stem

def build_paired_list(img_root: Path, mask_root: Path) -> List[Tuple[str, Path, Path]]:
    img_by_stem = list_files_by_stem(img_root)
    mask_by_stem = list_files_by_stem(mask_root)
    common_stems = sorted(set(img_by_stem.keys()) & set(mask_by_stem.keys()), key=str.lower)

    missing_img = sorted(set(mask_by_stem.keys()) - set(img_by_stem.keys()))
    missing_mask = sorted(set(img_by_stem.keys()) - set(mask_by_stem.keys()))
    if missing_img:
        print(f"[AVVISO] {len(missing_img)} maschere senza immagine in '{img_root.name}'.")
    if missing_mask:
        print(f"[AVVISO] {len(missing_mask)} immagini senza maschera in '{mask_root.name}'.")

    return [(stem, img_by_stem[stem], mask_by_stem[stem]) for stem in common_stems]

def split_indices(n: int, train_ratio: float, valid_ratio: float, test_ratio: float):
    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0):
        train_ratio, valid_ratio, test_ratio = train_ratio/total, valid_ratio/total, test_ratio/total
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    n_test = n - n_train - n_valid
    idx = list(range(n))
    return idx[:n_train], idx[n_train:n_train+n_valid], idx[n_train+n_valid:]

def ensure_clean_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def place_one(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if USE_SYMLINKS:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dest)

def place_pairs(
    pairs: List[Tuple[str, Path, Path]],
    out_img_dir: Path, out_mask_dir: Path,
    img_root: Path, mask_root: Path
):
    for _, img_p, mask_p in pairs:
        rel_img = img_p.relative_to(img_root)
        rel_mask = mask_p.relative_to(mask_root)
        place_one(img_p, out_img_dir / rel_img)
        place_one(mask_p, out_mask_dir / rel_mask)

# ===================== RANDOM ARG BUILDER =====================

def random_arms_str(include_triplet: bool = True) -> str:
    """
    Ritorna una combinazione valida in stringa: "1", "2", "3", "1 2", "1 3", "2 3", (opz) "1 2 3".
    """
    arms = [1, 2, 3]
    # scegli cardinalità: 1 o 2, e opzionalmente 3
    sizes = [1, 2] + ([3] if include_triplet else [])
    k = random.choice(sizes)
    pick = sorted(random.sample(arms, k))
    return " ".join(str(x) for x in pick)

def build_random_runs(n: int, seconds: str = "8") -> List[dict]:
    """
    Crea n run casuali con arms validi e seed random.
    """
    runs = []
    seen: Set[tuple] = set()
    tries = 0
    # provo a evitare duplicati grossolani (stessa coppia arms+seed)
    while len(runs) < n and tries < n * 20:
        tries += 1
        arms = random_arms_str(include_triplet=True)
        seed = random.randint(1, 2_147_483_647)
        key = (arms, seed)
        if key in seen:
            continue
        seen.add(key)
        runs.append({"arms": arms, "seconds": seconds, "seed": seed})
    return runs

# ===================== PIPELINE SINTETICI (ITERATIVA) =====================

def run_generation_pipeline(num_runs: int):
    """
    Esegue:
      - video_generation.py num_runs volte (args casuali)
      - batch_chromakey.py
      - estrai_frame_coppie.py
    cwd=UTILS_DIR
    """
    if not UTILS_DIR.exists():
        print(f"[ERRORE] Utils dir non trovata: {UTILS_DIR.resolve()}")
        return

    # 1) Video generation (mayapy) con run random
    gen_runs = build_random_runs(max(num_runs, MIN_RUNS_PER_ITER))
    ok = 0
    for run in gen_runs:
        try:
            cmd = [
                MAYAPY, "video_generation.py",
                run["arms"],        # es. "1 2"
                run["seconds"],     # "8"
                str(run["seed"]),   # seed random
            ]
            print(f"[INFO] Avvio: {' '.join(cmd)}  (cwd={UTILS_DIR})")
            subprocess.run(cmd, cwd=str(UTILS_DIR), check=True)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"[ERRORE] video_generation.py fallito (arms='{run['arms']}', seed={run['seed']}): {e}")
        except FileNotFoundError:
            print(f"[ERRORE] mayapy non trovato a '{MAYAPY}'. Aggiorna il percorso.")
            break

    if ok < MIN_RUNS_PER_ITER:
        print(f"[AVVISO] Solo {ok}/{MIN_RUNS_PER_ITER} run minime riuscite.")

    # 2) Chromakey
    chroma_cmd = [
        "python3", "batch_chromakey.py",
        "--in_dir", "/scratch/Tesi-Borra/OUTPUT/output_greenscreen",
        "--bg", "/scratch/Tesi-Borra/background.mp4",
        "--suffix", "",
        "--out_dir", "/Users/giorgiochiesa/Downloads/Borra_Codice_Tesi/Utils/OUTPUT/output_background",
        "--overwrite",
    ]
    try:
        print(f"[INFO] Avvio: {' '.join(chroma_cmd)}  (cwd={UTILS_DIR})")
        subprocess.run(chroma_cmd, cwd=str(UTILS_DIR), check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] batch_chromakey.py fallito: {e}")

    # 3) Estrai frame coppie
    estrai_cmd = ["python3", "estrai_frame_coppie.py"]
    try:
        print(f"[INFO] Avvio: {' '.join(estrai_cmd)}  (cwd={UTILS_DIR})")
        subprocess.run(estrai_cmd, cwd=str(UTILS_DIR), check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] estrai_frame_coppie.py fallito: {e}")

# ===================== PIPELINE PRINCIPALE =====================

def main():
    real_img = DATASET_DIR / "img"
    real_mask = DATASET_DIR / "mask"
    synth_img = DATASET_DIR / "sint_img"
    synth_mask = DATASET_DIR / "sint_mask"

    if not (real_img.exists() and real_mask.exists()):
        raise RuntimeError("Cartelle reali 'img' e/o 'mask' non trovate dentro DATASET_DIR.")
    if not (synth_img.exists() and synth_mask.exists()):
        print("[INFO] Cartelle sintetiche non trovate/completamente: userò solo reali per il train.")

    # Paia reali e sintetiche
    real_pairs = build_paired_list(real_img, real_mask)
    synth_pairs = build_paired_list(synth_img, synth_mask) if (synth_img.exists() and synth_mask.exists()) else []

    if len(real_pairs) == 0:
        raise RuntimeError("Nessuna coppia (img+mask) reale trovata.")

    # Split deterministico sui REALI
    train_idx, valid_idx, test_idx = split_indices(len(real_pairs), TRAIN_RATIO, VALID_RATIO, TEST_RATIO)
    real_train = [real_pairs[i] for i in train_idx]
    real_valid = [real_pairs[i] for i in valid_idx]
    real_test  = [real_pairs[i] for i in test_idx]

    # Quanti sintetici servono nel TRAIN
    alpha_eff = max(0.0, min(ALPHA, 1.0))
    if math.isclose(alpha_eff, 1.0, abs_tol=1e-9):
        alpha_eff = 0.99
    n_real_train = len(real_train)
    n_synth_needed = 0 if alpha_eff == 0 else int(math.floor((alpha_eff / (1 - alpha_eff)) * n_real_train))

    # --- Loop di generazione finché non raggiungiamo il fabbisogno ---
    if n_synth_needed > 0:
        iter_count = 0
        while len(synth_pairs) < n_synth_needed and iter_count < MAX_GENERATION_ITERS:
            deficit = (n_synth_needed - len(synth_pairs))/(24*8)
            # Numero run proporzionale al deficit, ma con un minimo e un tetto ragionevole
            runs_this_iter = 1#max(MIN_RUNS_PER_ITER, deficit)
            print(f"[INFO] Servono {n_synth_needed} sintetici, disponibili {len(synth_pairs)}. "
                  f"Lancio pipeline (iter {iter_count+1}) con {runs_this_iter} run video...")
            run_generation_pipeline(num_runs=runs_this_iter)
            # ri-scan dopo la pipeline
            synth_pairs = build_paired_list(synth_img, synth_mask)
            print(f"[INFO] Dopo iter {iter_count+1}, sintetici disponibili: {len(synth_pairs)}")
            iter_count += 1

        if len(synth_pairs) < n_synth_needed:
            print(f"[AVVISO] Raggiunto limite iterazioni ({MAX_GENERATION_ITERS}) ma servirebbero ancora "
                  f"{n_synth_needed - len(synth_pairs)} coppie sintetiche. Procedo con quelle disponibili.")

    # Selezione sintetici (deterministica)
    synth_selected = synth_pairs[:min(n_synth_needed, len(synth_pairs))]

    # ====== Scrittura output in DaVinci/ ======
    ensure_clean_dir(OUTPUT_DIR)

    out_train_img  = OUTPUT_DIR / "train_img"
    out_train_mask = OUTPUT_DIR / "train_mask"
    out_valid_img  = OUTPUT_DIR / "valid_img"
    out_valid_mask = OUTPUT_DIR / "valid_mask"
    out_test_img   = OUTPUT_DIR / "test_img"
    out_test_mask  = OUTPUT_DIR / "test_mask"

    # Reali: train/valid/test
    place_pairs(real_train, out_train_img, out_train_mask, real_img, real_mask)
    place_pairs(real_valid, out_valid_img, out_valid_mask, real_img, real_mask)
    place_pairs(real_test,  out_test_img,  out_test_mask,  real_img, real_mask)

    # Sintetici SOLO nel train
    if synth_selected:
        place_pairs(synth_selected, out_train_img, out_train_mask, synth_img, synth_mask)

    # ====== Riepilogo ======
    n_r_tr, n_r_val, n_r_te = len(real_train), len(real_valid), len(real_test)
    n_s_tr = len(synth_selected)
    total_train = n_r_tr + n_s_tr
    synth_ratio = (n_s_tr / total_train) if total_train > 0 else 0.0

    print("=== RIEPILOGO SPLIT ===")
    print(f"Reali: tot={len(real_pairs)}  -> train={n_r_tr}, valid={n_r_val}, test={n_r_te}")
    print(f"Sintetici disponibili/finali nel train: {len(synth_pairs)}/{n_s_tr}")
    print(f"TRAIN tot={total_train} | %synth={synth_ratio*100:.2f}%  (ALPHA={ALPHA}, effettiva usata={alpha_eff})")
    print(f"Output scritto in: {OUTPUT_DIR.resolve()}")

main()


# %%
DATA_DIR = "/scratch/Tesi-Borra/dataset/dataset/DaVinci"+str(conf.alpha)

x_train_dir = os.path.join(DATA_DIR, "train_img")
y_train_dir = os.path.join(DATA_DIR, "train_mask")

x_valid_dir = os.path.join(DATA_DIR, "valid_img")
y_valid_dir = os.path.join(DATA_DIR, "valid_mask")

x_test_dir = os.path.join(DATA_DIR, "test_img")
y_test_dir = os.path.join(DATA_DIR, "test_mask")

# %% [markdown]
# ## Data Loader

# %%
import os
import cv2
import numpy as np

class DatasetBinary(BaseDataset):
    """
    Dataset per segmentazione binaria.

    Args:
        images_dir (str): cartella immagini
        masks_dir (str): cartella maschere binarie (0/255 o 0/1)
        augmentation (albumentations.Compose, opzionale): pipeline di augmentations
        two_channels (bool): se True -> restituisce [background, foreground] (2 canali).
                             se False -> solo foreground (1 canale).
        threshold (int): soglia per binarizzare la mask letta (utile se è 0/255).
    """

    def __init__(self, images_dir, masks_dir, augmentation=None,
                 two_channels=False, threshold=0):
        # prendi solo i file che esistono in entrambe le cartelle con lo stesso nome
        img_names  = set(os.listdir(images_dir))
        mask_names = set(os.listdir(masks_dir))
        self.ids = sorted(list(img_names & mask_names))

        self.images_fps = [os.path.join(images_dir, f) for f in self.ids]
        self.masks_fps  = [os.path.join(masks_dir,  f) for f in self.ids]

        self.augmentation = augmentation
        self.two_channels = two_channels
        self.threshold = threshold

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
    # ---- immagine ----
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"[ERRORE] Immagine non trovata o corrotta: {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---- maschera (grayscale) ----
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask non trovata: {self.masks_fps[i]}")

    # binarizza
        mask = (mask > self.threshold).astype("float32")

        if self.two_channels:
            mask = np.stack([1.0 - mask, mask], axis=-1)
        else:
            mask = np.expand_dims(mask, axis=-1)

        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

    # (C,H,W)
        image = image.transpose(2, 0, 1).astype("float32")
        mask  = mask.transpose(2, 0, 1).astype("float32")

    # Ritorna anche i nomi dei file
        img_name  = os.path.basename(self.images_fps[i])
        mask_name = os.path.basename(self.masks_fps[i])

        return image, mask, img_name, mask_name



# %% [markdown]
# ## Visualize

# %%
def _to_hwc_uint8_or_float01(img):
    """Converte (C,H,W)->(H,W,C) e scala correttamente per imshow."""
    # porta a HWC
    if img.ndim == 3 and img.shape[0] in (1,3,4) and img.shape[0] < img.shape[1]:
        img = img.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)

    # se è float con valori >1, scala a [0,1]
    if np.issubdtype(img.dtype, np.floating):
        maxv = float(np.nanmax(img)) if np.isfinite(img).all() else 1.0
        if maxv > 1.0:
            img = img / 255.0  # assumiamo 0..255
        # altrimenti già 0..1 -> ok
    elif img.dtype != np.uint8:
        # altri interi -> converto a uint8 con clipping
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def visualize(image, mask, img_name=None, mask_name=None, two_channels=False):
    """Mostra immagine e maschera affiancate, con nomi file."""
    img_vis  = _to_hwc_uint8_or_float01(image)

    # mask: accetta (C,H,W) o (H,W) o (H,W,1) o 2 canali
    if mask.ndim == 3 and mask.shape[0] in (1,2):
        # (C,H,W) -> scegli canale foreground se two_channels, altrimenti il primo
        ch = 1 if (two_channels and mask.shape[0] == 2) else 0
        m = mask[ch]
    elif mask.ndim == 3 and mask.shape[-1] in (1,2):
        ch = 1 if (two_channels and mask.shape[-1] == 2) else 0
        m = mask[..., ch]
    else:
        m = mask

    # porta maschera a 2D
    m = np.squeeze(m)
    # normalizza mask per vis: 0/1 o 0..255 -> 0..1
    if np.issubdtype(m.dtype, np.floating):
        if m.max() > 1.0:
            m = m / 255.0
    else:
        m = m.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0

    plt.figure(figsize=(16,5))

    # immagine
    plt.subplot(1,2,1)
    plt.xticks([]); plt.yticks([])
    plt.title(f"Image: {img_name}" if img_name else "Image")
    plt.imshow(img_vis)

    # maschera
    plt.subplot(1,2,2)
    plt.xticks([]); plt.yticks([])
    plt.title(f"Mask: {mask_name}" if mask_name else "Mask")
    plt.imshow(m, cmap="gray", vmin=0, vmax=1)

    plt.show()


# %% [markdown]
# ## Augmentation

# %%

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True),
        A.RandomCrop(height=256, width=256, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    return A.Compose([
        A.Resize(256, 256)   # multiplo di 32
    ])
    return A.Compose(test_transform)

# %% [markdown]
# ## Datasets

# %%


train_dataset = DatasetBinary(
    x_train_dir, y_train_dir,
    augmentation=get_training_augmentation(),
    two_channels=False,  # o True se ti serve [bg, fg]
    threshold=0
)

valid_dataset = DatasetBinary(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    two_channels=False,  # o True se ti serve [bg, fg]
    threshold=0
)

test_dataset = DatasetBinary(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    two_channels=False,  # o True se ti serve [bg, fg]
    threshold=0
)

train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0)

# %% [markdown]
# ## DaVinciModel

# %%
# Some training hyperparameters
EPOCHS = conf.epochs
T_MAX = EPOCHS * len(train_loader)
OUT_CLASSES = 1

# %%



class DaVinciModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # preprocessing parameters per encoder
        params = smp.encoders.get_preprocessing_params(encoder_name)
        mean = torch.tensor(params["mean"], dtype=torch.float32)
        std  = torch.tensor(params["std"],  dtype=torch.float32)

        # se lavori in grayscale (in_channels=1) usa solo il primo canale di mean/std
        if in_channels == 1:
            mean = mean[:1]
            std  = std[:1]

        # shape (1,C,1,1) per broadcast
        self.register_buffer("mean", mean.view(1, in_channels, 1, 1))
        self.register_buffer("std",  std.view(1, in_channels, 1, 1))

        # dice loss binaria su logits
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # accumulatori per epoch-end
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # assicurati di avere float32
        if image.dtype != torch.float32:
            image = image.float()
        # normalizzazione per encoder
        image = (image - self.mean) / self.std
        return self.model(image)

    @staticmethod
    def _prepare_batch(batch):
        """
        Supporta:
          - (image, mask)
          - (image, mask, img_name, mask_name)
        E normalizza shape/dtype:
          - image: float32, (B,C,H,W)
          - mask:  float32, (B,1,H,W) in [0,1]
        Se mask ha 2 canali [bg, fg], tiene il canale fg.
        """
        # batch può essere una lista/tupla di tensors
        if len(batch) >= 2:
            image, mask = batch[0], batch[1]
        else:
            raise ValueError("Batch atteso con almeno (image, mask).")

        # cast
        image = image.float()
        mask = mask.float()

        # mask: (B,H,W) -> (B,1,H,W)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # mask: (B,2,H,W) -> prendi foreground (indice 1) e mantieni dimensione canale
        if mask.ndim == 4 and mask.shape[1] == 2:
            mask = mask[:, 1:2, ...]
        # clip di sicurezza
        mask = mask.clamp(0.0, 1.0)

        return image, mask

    def shared_step(self, batch, stage):
        image, mask = self._prepare_batch(batch)

        # controlli forma
        assert image.ndim == 4, f"image ndim atteso 4, trovato {image.ndim}"
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, "H e W devono essere multipli di 32"

        assert mask.ndim == 4 and mask.shape[1] == 1, f"mask shape attesa (B,1,H,W), trovata {tuple(mask.shape)}"
        assert mask.max() <= 1.0 and mask.min() >= 0.0, "mask deve essere in [0,1]"

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        # metriche
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs], dim=0)
        fp = torch.cat([x["fp"] for x in outputs], dim=0)
        fn = torch.cat([x["fn"] for x in outputs], dim=0)
        tn = torch.cat([x["tn"] for x in outputs], dim=0)

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou   = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        # se vuoi anche loss media in progress bar:
        if len(outputs) > 0 and "loss" in outputs[0]:
            mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
            metrics[f"{stage}_loss"] = mean_loss

        self.log_dict(metrics, prog_bar=True, on_epoch=True, logger=True)

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        # Lightning usa il valore associato a "loss" per il backward
        return out

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, "test")
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    


# %%
import os

def remove_ds_store(root_dir):
    """
    Rimuove tutti i file .DS_Store dentro root_dir e sottocartelle.
    """
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f == ".DS_Store":
                full_path = os.path.join(dirpath, f)
                try:
                    os.remove(full_path)
                    removed += 1
                    print(f"Rimosso: {full_path}")
                except Exception as e:
                    print(f"Errore nel rimuovere {full_path}: {e}")
    if removed == 0:
        print("Nessun .DS_Store trovato.")
    else:
        print(f"Totale file .DS_Store rimossi: {removed}")

# 🔧 Esempio di utilizzo:
remove_ds_store("/scratch/Tesi-Borra/dataset/train_img")  # sostituisci col percorso giusto
remove_ds_store("/scratch/Tesi-Borra/dataset/train_mask")
remove_ds_store("/scratch/Tesi-Borra/dataset/test_img")  # sostituisci col percorso giusto
remove_ds_store("/scratch/Tesi-Borra/dataset/test_mask")
remove_ds_store("/scratch/Tesi-Borra/dataset/valid_img")  # sostituisci col percorso giusto
remove_ds_store("/scratch/Tesi-Borra/dataset/valid_mask")


# %%
model = DaVinciModel("FPN", conf.architecture, in_channels=3, out_classes=OUT_CLASSES,encoder_weights=None)

# %% [markdown]
# ## Training

# %%
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    logger=wandb_logger,
    accelerator= conf.device,
    devices=1
)

train_metrics = trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
)

# %%
valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
run.log({"alpha":conf.alpha})

# %% [markdown]
# ## Results

# %%
# === Video dalle predizioni in memoria (test_loader + model) ===
# Raggruppa per suffisso dopo "_" (es. _0, _1), ordina i frame (0000, 0001, ...)
# e salva uno MP4 per gruppo.

import re
import cv2
import torch
import numpy as np
from pathlib import Path

# --------- CONFIG ----------
OUTPUT_DIR = Path("/scratch/Tesi-Borra/videos_out/_alpha"+str(conf.alpha))   # dove salvare i video
os.makedirs(OUTPUT_DIR, exist_ok=True)
FPS = 25                            # frame rate
MAKE_OVERLAY = True                 # True: overlay pred su immagine; False: video della sola maschera
aplha_color = 0.40                        # trasparenza overlay
USE_PROB_AS_ALPHA = False           # alpha proporzionale alla probabilità (sfumato) invece che binario
GROUPS_FILTER = None                # es. ["0"] per fare solo il video dei *_0 ; None = tutti i gruppi trovati
RESIZE_MISMATCHES = True            # ridimensiona eventuali frame fuori risoluzione
# ---------------------------

def to_rgb01(t):
    """(C,H,W) torch -> (H,W,3) numpy float32 in [0,1]."""
    x = t.detach().float().cpu().numpy()
    if x.ndim == 3:  # (C,H,W)
        x = x.transpose(1, 2, 0)
    if x.max() > 1.0:
        x = x / 255.0
    return x.astype(np.float32)

def to_mask01(t):
    """torch -> (H,W) numpy float32 in [0,1]. Accetta logits/prob o binaria con shape (1,H,W) o (H,W)."""
    m = t.detach().float().cpu().numpy()
    m = np.squeeze(m)
    if m.max() > 1.0:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0).astype(np.float32)

def rgba_overlay(mask01, color="blue", alpha=0.4, use_prob_alpha=False):
    """Ritorna (H,W,4) RGBA per overlay; alpha > 0 solo dove mask > 0."""
    h, w = mask01.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    colors = {
        "blue":  (0.0, 0.0, 1.0),
        "red":   (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "cyan":  (0.0, 1.0, 1.0),
        "mag":   (1.0, 0.0, 1.0),
        "yellow":(1.0, 1.0, 0.0),
    }
    r, g, b = colors.get("blue", (0.0, 0.0, 1.0))  # fisso blu per la pred
    overlay[..., 0] = r
    overlay[..., 1] = g
    overlay[..., 2] = b
    if use_prob_alpha:
        overlay[..., 3] = np.clip(mask01, 0, 1) * alpha
    else:
        overlay[..., 3] = (mask01 > 0.5).astype(np.float32) * alpha
    return overlay

def ensure_bgr(img):
    """Garantisce BGR uint8 per il writer video."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# pattern nomi file: 0000_0.png -> frame_idx=0, group_tag="0"
pattern = re.compile(r'^(\d+)_([^.]+)\.(png|jpg|jpeg|bmp|tif|tiff)$', re.IGNORECASE)

# bucket: group_tag -> list[(frame_idx:int, frame_bgr:np.ndarray)]
buckets = {}
h0 = w0 = None

# Device corrente del modello (fallback CPU)
device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
model.eval()

with torch.inference_mode():
    for batch in test_loader:
        # Il tuo test_loader produce (images, masks, nameimage, namemask)
        images, masks, nameimage, namemask = batch

        images = images.to(device, non_blocking=True)
        logits = model(images)
        pr_masks = torch.sigmoid(logits)

        for img_t, pr_t, ni, nm in zip(images, pr_masks, nameimage, namemask):
            # Prendo il nome dall'immagine; se non matcha, provo la mask
            fname = Path(str(ni)).name
            m = pattern.match(fname)
            if not m:
                fname = Path(str(nm)).name
                m = pattern.match(fname)
            if not m:
                print(f"[SKIP] nome non compatibile: {ni} / {nm}")
                continue

            frame_idx = int(m.group(1))
            group_tag = m.group(2)

            if GROUPS_FILTER is not None and group_tag not in set(GROUPS_FILTER):
                continue

            # Preparo il frame per il video
            img = to_rgb01(img_t)               # (H,W,3) float in [0,1]
            pr  = to_mask01(pr_t)               # (H,W) float in [0,1]

            if MAKE_OVERLAY:
                ov = rgba_overlay(
                    pr if USE_PROB_AS_ALPHA else (pr > 0.5).astype(np.float32),
                    color="blue",
                    alpha=aplha_color,
                    use_prob_alpha=USE_PROB_AS_ALPHA
                )
                a = ov[..., 3:4]                # (H,W,1)
                rgb = img * (1 - a) + ov[..., :3] * a
                frame = (rgb * 255.0)
                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                frame = ((pr > 0.5).astype(np.uint8) * 255)  # grigio 0/255

            frame = ensure_bgr(frame)

            # Risoluzione target dal primo frame
            if h0 is None or w0 is None:
                h0, w0 = frame.shape[:2]

            if (frame.shape[1], frame.shape[0]) != (w0, h0):
                if RESIZE_MISMATCHES:
                    frame = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_NEAREST)
                else:
                    print(f"[WARN] dimensioni diverse per {fname}, salto.")
                    continue

            buckets.setdefault(group_tag, []).append((frame_idx, frame))

# Scrittura video
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

if not buckets:
    print("Nessun frame raccolto: controlla pattern nomi o GROUPS_FILTER.")
else:
    for gtag, items in buckets.items():
        items.sort(key=lambda x: x[0])  # ordina per indice frame
        out_path = OUTPUT_DIR / (f"pred_overlay_group_{gtag}_{conf.dataset}_{conf.architecture}.mp4" if MAKE_OVERLAY
                                 else f"pred_mask_group_{gtag}_{conf.dataset}_{conf.architecture}.mp4")
        writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w0, h0), True)
        if not writer.isOpened():
            raise RuntimeError(f"Impossibile aprire VideoWriter su {out_path}")

        for _, frame in items:
            writer.write(frame)
        writer.release()
        print(f"[OK] gruppo {gtag}: {len(items)} frame -> {out_path}")



