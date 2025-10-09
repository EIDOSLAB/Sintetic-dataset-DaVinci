#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import maya.standalone
import maya.cmds as cmds
import sys, os, re, random, math

# ---------- CONFIG ----------
SCENE_PATH = "DaVinci_3D_Model.ma"
FPS_DEFAULT = 24
DURATION_DEFAULT = 5
ARM_ROOTS = ["Arm1", "Arm2", "Arm3"]
OUTPUT_BASENAME = "Arm_video"
GREEN_RGB = (0.0, 1.0, 0.0)
BG_SPHERE_NAME = "__BG_GreenSphere__"
# ---------------------------

# ---------- UTILS ----------
def parse_arms(arg: str):
    if not arg:
        return ["Arm1"]
    tokens = re.findall(r"[1-3]", arg)
    chosen = sorted(set(f"Arm{t}" for t in tokens), key=lambda n: int(n[-1]))
    return chosen or ["Arm1"]

def parse_int(arg, default):
    try: return int(arg)
    except: return default

def set_visibility(keep_list):
    for arm in ARM_ROOTS:
        if cmds.objExists(arm):
            cmds.setAttr(f"{arm}.visibility", 1 if arm in keep_list else 0)

def label_from_arms(arms): 
    return "_".join(arms)

# Cerca pinze sotto ArmN (tollerante a prefissi)
def find_pinza_under_arm(arm_root, pinza_idx):
    if not cmds.objExists(arm_root):
        return None
    target_exact = f"{arm_root}_{'pinza1' if pinza_idx==1 else 'pinza2'}"
    desc = cmds.listRelatives(arm_root, ad=True, path=True, type="transform") or []
    desc.append(cmds.ls(arm_root, long=True)[0])
    candidates = []
    for p in desc:
        short = p.split("|")[-1]
        if short == target_exact or short.endswith(f"_{'pinza1' if pinza_idx==1 else 'pinza2'}") or short == f"{'pinza1' if pinza_idx==1 else 'pinza2'}":
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda s: s.count("|"))
    return candidates[0]

# ---------- GREEN SCREEN ----------
def make_green_bg_sphere():
    if cmds.objExists(BG_SPHERE_NAME):
        try: cmds.delete(BG_SPHERE_NAME)
        except: pass
    sphere = cmds.polySphere(name=BG_SPHERE_NAME, r=1000, sx=64, sy=32)[0]
    try: cmds.polyNormal(sphere, normalMode=0, userNormalMode=0, ch=False)
    except: pass
    shd = cmds.shadingNode('surfaceShader', asShader=True, name="bgGreen_surf")
    sg  = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="bgGreen_SG")
    cmds.connectAttr(shd + ".outColor", sg + ".surfaceShader", force=True)
    r,g,b = GREEN_RGB
    cmds.setAttr(shd + ".outColor", r, g, b, type="double3")
    cmds.sets(sphere, e=True, forceElement=sg)
    shape = cmds.listRelatives(sphere, shapes=True, fullPath=True)[0]
    for a, v in {
        shape + ".castsShadows": 0,
        shape + ".receiveShadows": 0,
        shape + ".motionBlur": 0,
        shape + ".primaryVisibility": 1,
        shape + ".visibleInReflections": 0,
        shape + ".visibleInRefractions": 0,
        shape + ".doubleSided": 0,
    }.items():
        if cmds.objExists(a):
            try: cmds.setAttr(a, v)
            except: pass
    try:
        cmds.setAttr(shape + ".overrideEnabled", 1)
        cmds.setAttr(shape + ".overrideDisplayType", 2)
    except: pass
    return sphere

def create_panel_with_camera(cam):
    win = cmds.window(title="PlayblastWin", widthHeight=(960, 540))
    lay = cmds.paneLayout(configuration='single')
    panel = cmds.modelPanel(camera=cam, parent=lay)
    editor = cmds.modelPanel(panel, q=True, modelEditor=True)
    cmds.modelEditor(panel, e=True, grid=False, hud=False, manipulators=False,
                     displayAppearance='smoothShaded', wireframeOnShaded=False)
    cmds.showWindow(win)
    return win, panel, editor

# ---------- CURVE UTILS ----------

def clear_existing_keys(node, attr):
    if node and cmds.objExists(node):
        try:
            anim_curves = cmds.listConnections(f"{node}.{attr}", s=True, d=False, type="animCurve") or []
            for c in anim_curves:
                try: cmds.delete(c)
                except: pass
        except: pass

def set_key(node, attr, frame, value):
    cmds.setKeyframe(node, attribute=attr, t=frame, v=value)

def hermite_smooth(x): 
    return x*x*(3 - 2*x)

def sin_blend(seed_local, fps):
    random.seed(seed_local)
    # due sinusoidi a frequenze e fasi diverse
    w1 = 2*math.pi * random.uniform(0.15, 0.55) / fps
    w2 = 2*math.pi * random.uniform(0.03, 0.22) / fps
    phi1 = random.uniform(0, 2*math.pi)
    phi2 = random.uniform(0, 2*math.pi)
    def f(t):
        s1 = 0.5*(math.sin(w1*t + phi1) + 1.0)
        s2 = 0.5*(math.sin(w2*t + phi2) + 1.0)
        return hermite_smooth(0.65*s1 + 0.35*s2)  # 0..1 morbido
    return f

# ---------- ANIM PINZE ----------

def anim_scissors_generic(arm_root, axis_attr, max_deg_1, max_deg_2, seed, fps, duration_s):
    pinza1 = find_pinza_under_arm(arm_root, 1)
    pinza2 = find_pinza_under_arm(arm_root, 2)
    if not (pinza1 and pinza2):
        print(f"[{arm_root}] Pinze non trovate. Cercavo: {arm_root}_pinza1/_pinza2 o *pinza1/*pinza2")
        return

    frame_start = 1
    frame_end   = frame_start + duration_s * fps
    step = 2

    # curve “random ma stabili” derivate dal seed e dall'arm
    f1 = sin_blend((seed*997) ^ hash(arm_root) ^ 0x1A2B3C, fps)
    f2 = sin_blend((seed*1499) ^ hash(arm_root) ^ 0x4D5E6F, fps)

    for node in (pinza1, pinza2):
        clear_existing_keys(node, axis_attr)
        set_key(node, axis_attr, frame_start, 0.0)

    for f in range(frame_start+step, frame_end, step):
        t = f - frame_start
        v1 = f1(t) * max_deg_1                   # 0 .. +max
        v2 = -abs(f2(t) * abs(max_deg_2))        # 0 .. -|max|
        set_key(pinza1, axis_attr, f, v1)
        set_key(pinza2, axis_attr, f, v2)

    set_key(pinza1, axis_attr, frame_end, 0.0)
    set_key(pinza2, axis_attr, frame_end, 0.0)

    for node in (pinza1, pinza2):
        try: cmds.keyTangent(node, attribute=axis_attr, itt="auto", ott="auto")
        except: pass

# ---------- ANIM ROTAZIONI INTERO BRACCIO ----------

def anim_arm_rotations(arm_root, ranges_xyz, seed, fps, duration_s, base_offset=None):
    """
    ranges_xyz = dict con limiti per ciascun asse, es:
      {"rotateZ": (-20, 20), "rotateY": (-15, 15), "rotateX": (-5, 5)}
    Muove il transform del gruppo 'arm_root' ruotando nei limiti dati.

    base_offset = dict opzionale per offset costante (es. {"rotateY": -31}).
    Se presente, i key iniziale/finale e tutti i valori intermedi verranno shiftati dell'offset.
    """
    if not cmds.objExists(arm_root):
        return

    base_offset = base_offset or {}
    offZ = float(base_offset.get("rotateZ", 0.0))
    offY = float(base_offset.get("rotateY", 0.0))
    offX = float(base_offset.get("rotateX", 0.0))

    frame_start = 1
    frame_end   = frame_start + duration_s * fps
    step = 2

    # una funzione “sinusoidale mista” per ogni asse, derivata dallo stesso seed
    fZ = sin_blend((seed*1337) ^ (hash(arm_root) << 1) ^ 0xABC, fps)
    fY = sin_blend((seed*7331) ^ (hash(arm_root) << 2) ^ 0xDEF, fps)
    fX = sin_blend((seed*2025) ^ (hash(arm_root) << 3) ^ 0x123, fps)

    for axis_attr, off in (("rotateZ", offZ), ("rotateY", offY), ("rotateX", offX)):
        clear_existing_keys(arm_root, axis_attr)
        set_key(arm_root, axis_attr, frame_start, off)

    for f in range(frame_start+step, frame_end, step):
        t = f - frame_start
        # calcola valori per asse entro i range + offset
        for axis_attr, func, off in (("rotateZ", fZ, offZ), ("rotateY", fY, offY), ("rotateX", fX, offX)):
            mn, mx = ranges_xyz[axis_attr]
            v = mn + func(t) * (mx - mn) + off
            set_key(arm_root, axis_attr, f, v)

    # riportiamo ai valori di offset a fine clip
    for axis_attr, off in (("rotateZ", offZ), ("rotateY", offY), ("rotateX", offX)):
        set_key(arm_root, axis_attr, frame_end, off)
        try: cmds.keyTangent(arm_root, attribute=axis_attr, itt="auto", ott="auto")
        except: pass

# ---------- TRANSLATE UTILS ----------

def apply_translateX_delta_static(arm_root, delta, frame_start, frame_end):
    """Applica una traslazione RELATIVA di delta su translateX, statica per tutta la clip.
    Mantiene il valore per tutta la durata impostando key all'inizio e alla fine.
    Non tocca altri canali."""
    if not cmds.objExists(arm_root):
        return
    try:
        base = cmds.getAttr(f"{arm_root}.translateX")
    except:
        base = 0.0
    target = base + float(delta)
    clear_existing_keys(arm_root, "translateX")
    try:
        cmds.setAttr(f"{arm_root}.translateX", target)
    except:
        pass
    set_key(arm_root, "translateX", frame_start, target)
    set_key(arm_root, "translateX", frame_end, target)
    try:
        cmds.keyTangent(arm_root, attribute="translateX", itt="auto", ott="auto")
    except:
        pass
# ---------- BLOOD: per-shape, materiale random indipendente ----------

import hashlib

_BLOOD_REGEX = re.compile(r"^blood(_arm)?(_\d+)*$", re.IGNORECASE)  # blood, blood_arm, blood_arm_1_2, ecc.

def _short_no_ns(node):
    return node.split("|")[-1].split(":")[-1]

def _resolve_standard_surface(material_names):
    """Risolvi SOLO materiali 'standardSurface' (case/namespace-insensitive).
       Se arriva un nome di SG, risale al materiale collegato."""
    all_mats = cmds.ls(type="standardSurface") or []
    all_sgs  = cmds.ls(type="shadingEngine") or []
    targets = [n.split(":")[-1].lower() for n in material_names]
    found = []

    for m in all_mats:
        if _short_no_ns(m).lower() in targets:
            found.append(m)

    for sg in all_sgs:
        if _short_no_ns(sg).lower() in targets:
            src = cmds.listConnections(sg + ".surfaceShader", s=True, d=False) or []
            if src and cmds.nodeType(src[0]) == "standardSurface":
                found.append(src[0])

    # de-duplica preservando ordine
    out, seen = [], set()
    for m in found:
        if m not in seen:
            out.append(m); seen.add(m)
    return out

def _sg_for(mat):
    """Riuso/creo la shadingEngine per il materiale dato."""
    sgs = cmds.listConnections(mat, type="shadingEngine") or []
    sgs = [s for s in sgs if s != "initialShadingGroup"]
    if sgs:
        return sgs[0]
    base = re.sub(r"(?i)shape|sg$", "", _short_no_ns(mat))
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"{base}SG")
    plug = f"{mat}.outColor" if cmds.objExists(f"{mat}.outColor") else (f"{mat}.out" if cmds.objExists(f"{mat}.out") else None)
    if plug:
        try: cmds.connectAttr(plug, f"{sg}.surfaceShader", force=True)
        except: pass
    return sg

def _list_blood_shapes_under_arm(arm_root):
    """Tutte le SHAPE che stanno sotto qualunque transform 'blood_*' dell'arm dato."""
    if not cmds.objExists(arm_root): return []
    desc = cmds.listRelatives(arm_root, ad=True, path=True) or []
    desc += cmds.ls(arm_root, long=True) or []
    shapes = set()
    for tr in desc:
        short = tr.split("|")[-1]
        if _BLOOD_REGEX.match(short):
            # prendi sia le shape dirette sia quelle nei discendenti
            shps = cmds.listRelatives(tr, shapes=True, fullPath=True) or []
            shps += cmds.listRelatives(tr, ad=True, shapes=True, fullPath=True) or []
            for s in shps:
                shapes.add(s)
    # ordina per stabilità
    return sorted(shapes)

def _choose_mat_deterministic(mats, seed, shape_path):
    """Scelta stabile per shape (stesso seed -> stesso risultato).
       Usa md5 del nome completo per evitare il random-hash di Python."""
    h = int(hashlib.md5(shape_path.encode("utf-8")).hexdigest(), 16)
    idx = (h ^ (seed or 0)) % len(mats)
    return mats[idx]

def randomize_blood_materials_per_shape(arms, material_names, seed=None):
    """
    Per OGNI SHAPE sotto i transform 'blood_*' di ciascun Arm, assegna
    un materiale 'standardSurface' randomico indipendente dagli altri.
    Se seed è None -> random “vero” ad ogni run. Se seed è un int -> mapping stabile.
    """
    mats = _resolve_standard_surface(material_names)
    if not mats:
        print("[BLOOD] Nessun 'standardSurface' trovato tra:", material_names)
        return

    total_shapes = 0
    for arm in arms:
        shapes = _list_blood_shapes_under_arm(arm)
        for shp in shapes:
            if seed is None:
                mat = random.choice(mats)
            else:
                mat = _choose_mat_deterministic(mats, seed, shp)
            sg = _sg_for(mat)
            try:
                cmds.sets(shp, e=True, forceElement=sg)
                total_shapes += 1
            except:
                pass

    print(f"[BLOOD] Assegnato materiale random per-shape a {total_shapes} shape. Candidati: {[ _short_no_ns(m) for m in mats ]}")


# ---------- MAIN ----------

def main():
    # Inizializza Maya standalone
    try:
        maya.standalone.initialize(name='python')
    except:
        pass

    if not os.path.exists(SCENE_PATH):
        cmds.error(f"File non trovato: {SCENE_PATH}")
        return

    cmds.file(SCENE_PATH, open=True, force=True)

    # CLI: bracci, durata, seed
    arms_arg   = sys.argv[1] if len(sys.argv) > 1 else ""
    duration_s = parse_int(sys.argv[2], DURATION_DEFAULT) if len(sys.argv) > 2 else DURATION_DEFAULT
    seed       = parse_int(sys.argv[3], 1234) if len(sys.argv) > 3 else 1234
    fps        = FPS_DEFAULT

    chosen_arms = parse_arms(arms_arg)
    chosen_arms = [a for a in chosen_arms if cmds.objExists(a)]
    if not chosen_arms:
        if cmds.objExists("Arm1"):
            chosen_arms = ["Arm1"]
        else:
            cmds.error("Nessun braccio trovato (Arm1/Arm2/Arm3).")
            return

    # Visibilità
    set_visibility(chosen_arms)
        # --- Materiale random per i blood_arm_1 ---
    randomize_blood_materials_per_shape(
    chosen_arms,
    material_names=["standardSurface1", "standardSurface2", "standardSurface3","standardSurface4","standardSurface5"],
    seed=seed  # oppure None se vuoi risultati diversi a ogni run
)




    # --- Offset Y statico se sono esattamente 2 bracci ---
    # Il primo (in chosen_arms, già ordinati) -> -31° su Y; il secondo -> +31° su Y
    two_arm_offsets = {}
    if len(chosen_arms) == 2:
        two_arm_offsets[chosen_arms[0]] = {"rotateY": -31.0}
        two_arm_offsets[chosen_arms[1]] = {"rotateY": +31.0}

    # --- Traslazioni su X per Arm2 quando scelgo 2 bracci ---
    frame_start = 1
    frame_end   = frame_start + duration_s * fps
    if len(chosen_arms) == 2:
        pair = set(chosen_arms)
        if pair == {"Arm1", "Arm2"}:
            # Arm2 -> -3 su X (relativo)
            apply_translateX_delta_static("Arm2", -4.0, frame_start, frame_end)
        elif pair == {"Arm2", "Arm3"}:
            # Arm2 -> +3 su X (relativo)
            apply_translateX_delta_static("Arm2", +4.0, frame_start, frame_end)
        # pair == {"Arm1", "Arm3"} -> nessuna traslazione
    # Se 3 bracci selezionati o uno solo -> non cambia nulla

    # --- ANIM PINZE & ROTAZIONI BRACCIO ---
    for arm in chosen_arms:
        idx = int(arm[-1])

        # Pinze
        if arm == "Arm1":
            anim_scissors_generic("Arm1", "rotateY", 20.0, -14.0, seed + idx*1000, fps, duration_s)
            ranges = {"rotateZ": (-50, 50), "rotateY": (-15, 15), "rotateX": (-5, 5)}
        elif arm == "Arm2":
            anim_scissors_generic("Arm2", "rotateY", 60.0, -60.0, seed + idx*1000, fps, duration_s)
            ranges = {"rotateZ": (-20, 20), "rotateY": (-12, 12), "rotateX": (-5, 5)}
        elif arm == "Arm3":
            anim_scissors_generic("Arm3", "rotateX", 80.0, -80.0, seed + idx*1000, fps, duration_s)
            ranges = {"rotateZ": (-30, 30), "rotateY": (-6, 6), "rotateX": (-5, 5)}
        else:
            continue

        # Rotazioni del gruppo ArmN secondo i range richiesti + eventuale offset Y
        base_off = two_arm_offsets.get(arm, None)
        anim_arm_rotations(arm, ranges, seed, fps, duration_s, base_offset=base_off)

    # Camera + inquadratura
    cam = cmds.camera(name="videoCam")[0]
    cmds.select(chosen_arms, r=True)
    cmds.viewFit(cam)

    # Green screen geometrico
    make_green_bg_sphere()

    # Pannello dedicato
    win, panel, editor = create_panel_with_camera(cam)

    # Timeline
    cmds.playbackOptions(min=frame_start, max=frame_end)

    # Output

# cartella in cui si trova lo script
    script_dir = os.path.dirname(os.path.abspath(__file__))

# cartella OUTPUT/output_script1 dentro la cartella dello script
    documents = os.path.join(script_dir, "OUTPUT", "output_greenscreen")

# crea la cartella se non esiste
    os.makedirs(documents, exist_ok=True)

# nome file finale
    label = label_from_arms(chosen_arms)
    outfile = os.path.join(documents, f"{OUTPUT_BASENAME}_{label}_{duration_s}s_seed{seed}.mp4")


    # Playblast MP4 H.264
    cmds.playblast(
        startTime=frame_start,
        endTime=frame_end,
        filename=outfile,
        format="movie",
        compression="H.264",
        forceOverwrite=True,
        viewer=False,
        showOrnaments=False,
        percent=100,
        width=1920,
        height=1080,
        offScreen=True
    )

    print(f"Video salvato in: {outfile}")

    try: cmds.deleteUI(win)
    except: pass
    try: maya.standalone.uninitialize()
    except: pass

if __name__ == "__main__":
    main()
