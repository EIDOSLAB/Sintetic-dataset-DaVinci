#!/usr/bin/env python3
import cv2
import sys
import re
import shutil
from pathlib import Path

# ---- Parametri HSV per il rilevamento del green screen ----
HSV_LOWER = (35, 40, 40)   # H: ~35-85 circa è il verde; S/V evitano grigi/neri
HSV_UPPER = (85, 255, 255)

# --- Percorsi relativi alla posizione di questo script ---
SCRIPT_DIR = Path(__file__).resolve().parent
# cartella padre (accanto a Utils/)
PARENT_DIR = SCRIPT_DIR.parent

# cartelle dataset richieste
DATASET_IMG_DIR  = (PARENT_DIR / "dataset" / "dataset" / "sint_img").resolve()   # output immagini con background
DATASET_MASK_DIR = (PARENT_DIR / "dataset" / "dataset" / "sint_mask").resolve()  # output maschere

# cartella da cui leggere il max indice esistente (le immagini esistenti con background)
REFERENCE_DIR = DATASET_IMG_DIR

# cartelle di archivio fisse (relative al progetto)
ARCHIVE_BG_DIR = Path("./OUTPUT/archive/output_background").resolve()
ARCHIVE_GS_DIR = Path("./OUTPUT/archive/output_greenscreen").resolve()

FILENAME_RE = re.compile(r"^(\d{4,})_data_sint\.png$", re.IGNORECASE)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pair_videos(bg_dir: Path, gs_dir: Path):
    """Ritorna lista di tuple (stem, bg_path, gs_path) per file con stesso stem."""
    bg_map = {p.stem: p for p in bg_dir.iterdir() if p.is_file()}
    gs_map = {p.stem: p for p in gs_dir.iterdir() if p.is_file()}
    common = sorted(set(bg_map.keys()) & set(gs_map.keys()))
    return [(stem, bg_map[stem], gs_map[stem]) for stem in common]

def get_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else 0

def estimate_total_frames(pairs) -> int:
    total = 0
    for _, bg_path, gs_path in pairs:
        n_bg = get_frame_count(bg_path)
        n_gs = get_frame_count(gs_path)
        if n_bg == 0 or n_gs == 0:
            continue
        total += min(n_bg, n_gs)
    return total

def green_to_binary_mask(frame_bgr):
    """Ritorna maschera 8-bit: verde -> 0 (nero), resto -> 255 (bianco)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)  # 255 per verde
    # Piccola pulizia morfologica per eliminare rumore
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Inverti: verde->0, resto->255
    mask_inv = cv2.bitwise_not(mask_green)
    return mask_inv  # singolo canale 8-bit

def find_start_index(reference_dir: Path) -> int:
    """Trova il max indice in reference_dir e ritorna max+1. Se non trova nulla -> 1."""
    if not reference_dir.is_dir():
        return 1
    max_num = 0
    for p in reference_dir.iterdir():
        if not p.is_file():
            continue
        m = FILENAME_RE.match(p.name)
        if m:
            try:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return max_num + 1 if max_num > 0 else 1

def _unique_target_path(dst_dir: Path, name: str) -> Path:
    """Se name esiste in dst_dir, aggiunge suffisso _1, _2, ... prima dell'estensione."""
    base = Path(name)
    stem, suffix = base.stem, base.suffix
    candidate = dst_dir / name
    k = 1
    while candidate.exists():
        candidate = dst_dir / f"{stem}_{k}{suffix}"
        k += 1
    return candidate

def _safe_move(src: Path, dst_dir: Path):
    """Sposta src in dst_dir, creando cartelle e gestendo collisioni."""
    ensure_dir(dst_dir)
    target = _unique_target_path(dst_dir, src.name)
    try:
        shutil.move(str(src), str(target))
    except Exception as e:
        print(f"[ATTENZIONE] Impossibile archiviare '{src}': {e}")

def process_pairs(pairs, out_bg_root: Path, out_gs_root: Path, start_index: int, pad: int,
                  archive_bg_dir: Path, archive_gs_dir: Path):
    """Scrive i PNG con contatore globale che parte da start_index e archivia i video elaborati."""
    index = start_index
    for stem, bg_path, gs_path in pairs:
        cap_bg = cv2.VideoCapture(str(bg_path))
        cap_gs = cv2.VideoCapture(str(gs_path))

        if not cap_bg.isOpened():
            print(f"[ERRORE] Non apro background: {bg_path}")
            # non archivio se non ho aperto il file
            continue
        if not cap_gs.isOpened():
            print(f"[ERRORE] Non apro greenscreen: {gs_path}")
            cap_bg.release()
            # non archivio se non ho aperto il file
            continue

        w_bg = int(cap_bg.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_bg = int(cap_bg.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_gs = int(cap_gs.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_gs = int(cap_gs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w_bg, h_bg) != (w_gs, h_gs):
            print(f"[ATTENZIONE] Dimensioni diverse per '{stem}': "
                  f"BG={w_bg}x{h_bg}, GS={w_gs}x{h_gs}. Salvo comunque in parallelo.")

        wrote_any = False

        while True:
            ret_bg, frame_bg = cap_bg.read()
            ret_gs, frame_gs = cap_gs.read()
            if not ret_bg or not ret_gs:
                if ret_bg != ret_gs:
                    print(f"[ATTENZIONE] Lunghezze diverse per '{stem}'. Fermato a index {index}.")
                break

            # Nome comune (es. 0123_data_sint.png)
            fname = f"{str(index).zfill(pad)}_data_sint.png"
            path_bg   = out_bg_root / fname   # immagini (background)
            path_mask = out_gs_root / fname   # maschere

            ok1 = cv2.imwrite(str(path_bg), frame_bg)
            mask = green_to_binary_mask(frame_gs)
            ok2 = cv2.imwrite(str(path_mask), mask)

            if not ok1 or not ok2:
                print(f"[ERRORE] Scrittura fallita a index {index} ('{stem}').")
                break

            wrote_any = True
            index += 1

        cap_bg.release()
        cap_gs.release()

        # Archivia i video dopo l'elaborazione (anche parziale)
        if wrote_any:
            _safe_move(bg_path, archive_bg_dir)
            _safe_move(gs_path, archive_gs_dir)
        else:
            print(f"[INFO] Nessun frame scritto per '{stem}': non archivio i file sorgente.")

    print("[OK] Completato.")

def main():
    # Uso:
    #   python estrai_frame_coppie.py
    #   python estrai_frame_coppie.py input_background input_greenscreen
    if len(sys.argv) not in (1, 3):
        print("Uso:\n  python estrai_frame_coppie.py [input_background input_greenscreen]")
        sys.exit(1)

    if len(sys.argv) == 3:
        bg_dir = Path(sys.argv[1])
        gs_dir = Path(sys.argv[2])
    else:
        # default: cartelle di input accanto al progetto, modificale se vuoi
        bg_dir = Path("./OUTPUT/output_background")
        gs_dir = Path("./OUTPUT/output_greenscreen")

    if not bg_dir.is_dir():
        print(f"[ERRORE] Cartella background non trovata: {bg_dir}")
        sys.exit(1)
    if not gs_dir.is_dir():
        print(f"[ERRORE] Cartella greenscreen non trovata: {gs_dir}")
        sys.exit(1)

    # Output fissi nelle cartelle dataset richieste
    out_bg_root  = DATASET_IMG_DIR
    out_gs_root  = DATASET_MASK_DIR

    # Crea tutte le cartelle necessarie (output e archivio)
    ensure_dir(out_bg_root)
    ensure_dir(out_gs_root)
    ensure_dir(ARCHIVE_BG_DIR)
    ensure_dir(ARCHIVE_GS_DIR)

    # Coppie di video da processare
    pairs = pair_videos(bg_dir, gs_dir)
    if not pairs:
        print("[ATTENZIONE] Nessuna coppia di video con lo stesso nome trovata.")
        sys.exit(0)

    # Indice iniziale guardando le immagini già presenti in sint_img
    start_index = find_start_index(REFERENCE_DIR)

    # Stima del totale per definire il padding (almeno 4)
    total = estimate_total_frames(pairs)
    pad = max(4, len(str(start_index + total))) if total > 0 else max(4, len(str(start_index)))

    print(f"Trovate {len(pairs)} coppie. Frame stimati: {total}. "
          f"Indice iniziale: {start_index}. Padding: {pad}.")
    print(f"Salvo IMG in:  {out_bg_root}")
    print(f"Salvo MASK in: {out_gs_root}")
    print(f"Archivio BG in:  {ARCHIVE_BG_DIR}")
    print(f"Archivio GS in:  {ARCHIVE_GS_DIR}")

    process_pairs(pairs, out_bg_root, out_gs_root, start_index, pad,
                  ARCHIVE_BG_DIR, ARCHIVE_GS_DIR)

if __name__ == "__main__":
    main()
