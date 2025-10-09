#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Sostituisce il green screen di TUTTI i video in una cartella con un video di background."
    )
    # modalità batch
    ap.add_argument("--in_dir", required=True, help="Cartella con i video foreground (green screen).")
    ap.add_argument("--bg", required=True, help="Video di background (riutilizzato per tutti).")
    ap.add_argument("--out_dir", required=True, help="Cartella di output.")
    ap.add_argument("--suffix", default="_ck", help="Suffisso aggiunto al nome file di output (default: _ck).")
    ap.add_argument("--ext", nargs="+", default=["mp4", "mov", "avi", "mkv", "m4v"],
                    help="Estensioni video da processare (senza punto). Default: mp4 mov avi mkv m4v")
    ap.add_argument("--overwrite", action="store_true", help="Sovrascrive se l'output esiste già.")

    # parametri chroma key (come nel tuo script)
    ap.add_argument("--lower", nargs=3, type=int, default=[35, 40, 40],
                    help="Soglia HSV inferiore per il verde (default: 35 40 40).")
    ap.add_argument("--upper", nargs=3, type=int, default=[85, 255, 255],
                    help="Soglia HSV superiore per il verde (default: 85 255 255).")
    ap.add_argument("--feather", type=int, default=7,
                    help="Raggio sfocatura/feathering della maschera (odd, default: 7).")
    ap.add_argument("--morph", type=int, default=3,
                    help="Kernel (px) per apertura/chiusura morfologica (default: 3).")
    ap.add_argument("--spill", type=float, default=0.15,
                    help="Green spill suppression (0–1, 0 = off, default: 0.15).")
    ap.add_argument("--preview", action="store_true",
                    help="Mostra anteprima durante l'elaborazione.")
    return ap

def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {path}")
    return cap

def process_one(fg_path: Path, bg_path: Path, out_path: Path, args):
    cap_fg = open_video(fg_path)
    cap_bg = open_video(bg_path)

    # Proprietà del video foreground (usate per output)
    width  = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap_fg.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0  # fallback

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # compatibile con .mp4
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap_fg.release()
        cap_bg.release()
        raise RuntimeError(f"Impossibile aprire l'output per la scrittura: {out_path}")

    lower = np.array(args.lower, dtype=np.uint8)
    upper = np.array(args.upper, dtype=np.uint8)
    feather = max(1, args.feather | 1)  # forza valore dispari
    morph_k = max(1, args.morph)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))

    # ciclo sui frame
    while True:
        ok_fg, frame_fg = cap_fg.read()
        if not ok_fg:
            break

        # Leggi/cicla BG e ridimensiona a misura FG
        ok_bg, frame_bg = cap_bg.read()
        if not ok_bg:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_bg, frame_bg = cap_bg.read()
            if not ok_bg:
                # se anche riavvolgendo fallisce, interrompo
                break

        frame_bg = cv2.resize(frame_bg, (width, height), interpolation=cv2.INTER_CUBIC)

        # --- CHROMA KEY ---
        hsv = cv2.cvtColor(frame_fg, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)  # 255 = verde (da sostituire)

        # Pulisci maschera (rumore/buchi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Feather del bordo per transizione morbida
        mask_feather = cv2.GaussianBlur(mask, (feather, feather), 0)

        # Inverti per avere alpha del soggetto
        alpha = 1.0 - (mask_feather.astype(np.float32) / 255.0)
        alpha_3 = cv2.merge([alpha, alpha, alpha])

        # Spill suppression (riduci dominante verde vicino ai bordi)
        if args.spill > 0:
            border = cv2.Canny(mask, 50, 150)
            border = cv2.GaussianBlur(border, (feather, feather), 0) / 255.0
            border = border[..., None]
            fg_f = frame_fg.astype(np.float32)
            rb_mean = (fg_f[..., 2] + fg_f[..., 0]) / 2.0
            g_corr = fg_f[..., 1] * (1.0 - args.spill * border[..., 0]) + rb_mean * (args.spill * 0.5 * border[..., 0])
            fg_f[..., 1] = g_corr
            frame_fg = np.clip(fg_f, 0, 255).astype(np.uint8)

        # Compositing: out = alpha * FG + (1 - alpha) * BG
        comp = (alpha_3 * frame_fg.astype(np.float32) + (1.0 - alpha_3) * frame_bg.astype(np.float32))
        comp = np.clip(comp, 0, 255).astype(np.uint8)

        out.write(comp)

        if args.preview:
            cv2.imshow("Compositing preview", comp)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
                break

    cap_fg.release()
    cap_bg.release()
    out.release()
    if args.preview:
        cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    bg_path = Path(args.bg).expanduser().resolve()

    if not in_dir.is_dir():
        raise RuntimeError(f"Cartella input non valida: {in_dir}")
    if not bg_path.is_file():
        raise RuntimeError(f"Video background non trovato: {bg_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.lower().lstrip(".") for e in args.ext}
    videos = [p for p in sorted(in_dir.iterdir())
              if p.is_file() and p.suffix.lower().lstrip(".") in exts and not p.name.startswith(".")]

    if not videos:
        print("Nessun video trovato nella cartella di input con le estensioni specificate.")
        return

    print(f"Trovati {len(videos)} video. Inizio elaborazione...\n")

    for idx, fg_path in enumerate(videos, 1):
        out_name = f"{fg_path.stem}{args.suffix}{fg_path.suffix}"
        out_path = out_dir / out_name

        if out_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(videos)}] Salto (esiste già): {out_path}")
            continue

        try:
            print(f"[{idx}/{len(videos)}] {fg_path.name}  ->  {out_name}")
            process_one(fg_path, bg_path, out_path, args)
            print(f"   ✔ Fatto: {out_path}")
        except Exception as e:
            print(f"   ✖ Errore su {fg_path.name}: {e}")

    print("\nCompletato. Output in:", out_dir)

if __name__ == "__main__":
    main()
