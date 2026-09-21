"""OpenThai-SystemOne plays Doom (ViZDoom), no vision, no text generation.

Every decision step the engine's symbolic state (health, ammo, visible enemies/items with distance + bearing, wall depth,
recent actions) is serialised to text and the model answers, in ONE forward pass:
  - choice: which action next (7 options)                 -> the key we press
  - noul:   is an enemy in the crosshair right now        -> shown as a threat bar
The clip shows the game on the left and, on the right, the live state, the probabilities, latency and decisions/sec.

    CUDA_VISIBLE_DEVICES=1 python demo/doom_demo.py --model release/OpenThai-SystemOne-v0.2 --out runs/demo/doom.mp4 --seconds 75
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openthai_systemone import SystemOneClient, Choice, Noul  # noqa: E402

import vizdoom as vzd  # noqa: E402

ACTIONS = {  # option name -> (description, button)
    "MOVE_FORWARD": ("run forward along the corridor", vzd.Button.MOVE_FORWARD),
    "TURN_LEFT": ("turn the view left", vzd.Button.TURN_LEFT),
    "TURN_RIGHT": ("turn the view right", vzd.Button.TURN_RIGHT),
    "ATTACK": ("fire the weapon at what is in the crosshair", vzd.Button.ATTACK),
    "MOVE_LEFT": ("strafe left without turning", vzd.Button.MOVE_LEFT),
    "MOVE_RIGHT": ("strafe right without turning", vzd.Button.MOVE_RIGHT),
    "MOVE_BACKWARD": ("step back", vzd.Button.MOVE_BACKWARD),
}
ENEMY_NAMES = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "Imp", "Demon", "Spectre", "Cacodemon", "LostSoul", "HellKnight", "BaronOfHell", "Revenant", "Arachnotron", "Mancubus"}
POLICY = ("Rules: if an enemy is VISIBLE and its bearing is within 8 degrees, ATTACK. If a VISIBLE enemy is to the left (positive "
          "bearing) TURN_LEFT; to the right (negative bearing) TURN_RIGHT. If no enemy is visible, {goal}. If STUCK, turn or strafe.")
SCENARIOS = {  # scenario -> (goal sentence, situation sentence)
    "deadly_corridor": ("MOVE_FORWARD toward the green armor at the end of the corridor", "You control a Doom marine in a corridor; enemies shoot from alcoves on both sides; a green armor at the far end is the goal."),
    "defend_the_center": ("keep scanning by turning", "You stand in the middle of a circular arena; monsters approach from all directions; survive and kill as many as possible."),
    "defend_the_line": ("keep scanning by turning", "You stand at one end of a hall; monsters spawn at the far end and advance; kill them before they reach you."),
    "health_gathering": ("MOVE_FORWARD toward the nearest Medikit, turning toward it first", "You are losing health on acid floor; medikits restore it; walk over as many medikits as you can."),
    "basic": ("MOVE_LEFT or MOVE_RIGHT to line the monster up with the crosshair", "A single monster stands in front of you; strafe until it is centered, then fire."),
    "my_way_home": ("MOVE_FORWARD through rooms toward the green armor, turning when a wall is ahead", "You are in a maze of rooms; find the green armor."),
    "take_cover": ("MOVE_LEFT or MOVE_RIGHT to dodge fireballs, keep changing side", "Monsters throw fireballs at you from the far end; you cannot fight, only dodge sideways."),
}
THREAT_Q = "Is an enemy inside the crosshair (bearing within 6 degrees, in view) so that firing now would hit it?"
HUD = {
    "en": {"title": "OpenThai-SystemOne  ·  Doom", "sub": "0.8B decision model · no vision · no text generation · 1 pass per step",
           "state": "STATE (what the model reads)", "choice": "CHOICE · next action", "noul": "NOUL · enemy in crosshair?", "pyes": "P(yes)",
           "lat": "latency {lat:4.0f} ms  ·  {dps:3.1f} decisions/s  ·  step {step}", "stats": "health {hp:.0f}   kills {kills:.0f}   output tokens: 0",
           "foot": "huggingface.co/iapp/OpenThai-SystemOne · Apache-2.0"},
    "th": {"title": "OpenThai-SystemOne  ·  Doom", "sub": "โมเดลตัดสินใจ 0.8B · ไม่ใช้ภาพ · ไม่สร้างข้อความ · 1 forward pass ต่อก้าว",
           "state": "STATE · สิ่งที่โมเดลอ่าน", "choice": "CHOICE · เลือกการกระทำถัดไป", "noul": "NOUL · มีศัตรูอยู่ในเป้าหรือไม่", "pyes": "P(ใช่)",
           "lat": "หน่วง {lat:4.0f} ms  ·  {dps:3.1f} ครั้ง/วินาที  ·  ก้าวที่ {step}", "stats": "พลังชีวิต {hp:.0f}   สังหาร {kills:.0f}   โทเคนที่สร้าง: 0",
           "foot": "huggingface.co/iapp/OpenThai-SystemOne · Apache-2.0 · iApp / OpenThaiGPT"},
}


def rel_bearing(px, py, pa, ox, oy):
    ang = math.degrees(math.atan2(oy - py, ox - px)) - pa
    return (ang + 180) % 360 - 180


def describe_state(game, state, history, stuck):
    hp = game.get_game_variable(vzd.GameVariable.HEALTH)
    ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
    kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    player = next((o for o in state.objects if o.name == "DoomPlayer"), None)
    visible = {lab.object_id for lab in (state.labels or [])}
    enemies, items = [], []
    if player is not None:
        for o in state.objects:
            if o.name == "DoomPlayer":
                continue
            d = math.hypot(o.position_x - player.position_x, o.position_y - player.position_y) / 32.0  # ~metres
            b = rel_bearing(player.position_x, player.position_y, player.angle, o.position_x, o.position_y)
            side = "ahead" if abs(b) < 15 else ("left" if b > 0 else "right")
            if o.name in ENEMY_NAMES:
                enemies.append((d, f"{o.name} {d:.0f}m {side} {b:+.0f}deg{' VISIBLE' if o.id in visible else ''}"))
            elif "Armor" in o.name or "Medikit" in o.name or "Stimpack" in o.name:
                items.append((d, f"{o.name} {d:.0f}m {side} {b:+.0f}deg"))
    enemies.sort(); items.sort()
    depth = state.depth_buffer
    wall = ""
    if depth is not None:
        h, w = depth.shape
        centre = float(depth[h // 2 - 5 : h // 2 + 5, w // 2 - 10 : w // 2 + 10].mean())
        wall = f"depth ahead: {centre:.0f}/255 ({'wall very close' if centre < 12 else 'open' if centre > 40 else 'obstacle near'})"
    aim = "crosshair: empty"
    if enemies:
        d0, e0 = enemies[0]
        b0 = float(e0.split("deg")[0].split()[-1])
        vis0 = "VISIBLE" in e0
        if vis0 and abs(b0) <= 8:
            aim = f"crosshair: ON {e0.split()[0]} ({b0:+.0f}deg) -> ATTACK would hit"
        else:
            aim = f"crosshair: empty; nearest {'visible ' if vis0 else ''}enemy {abs(b0):.0f}deg to the {'LEFT' if b0 > 0 else 'RIGHT'}"
    lines = [f"health {hp:.0f}/100 | ammo {ammo:.0f} | kills {kills:.0f}",
             aim,
             "enemies: " + ("; ".join(e for _, e in enemies[:4]) if enemies else "none in range"),
             "items: " + ("; ".join(i for _, i in items[:2]) if items else "none"),
             wall,
             "last actions: " + (" ".join(history) if history else "none") + (" | STUCK: position unchanged" if stuck else "")]
    return "\n".join(l for l in lines if l), enemies


def draw_panel(draw, fonts, hud, x0, w, h, state_text, probs, chosen, threat, latency_ms, dps, step, hp, kills):
    f_h, f_m, f_s = fonts
    pad = 26
    draw.rectangle([x0, 0, x0 + w, h], fill=(18, 18, 24))
    draw.text((x0 + pad, 26), hud["title"], font=f_h, fill=(255, 255, 255))
    draw.text((x0 + pad, 74), hud["sub"], font=f_s, fill=(150, 150, 165))
    y = 130
    draw.text((x0 + pad, y), hud["state"], font=f_m, fill=(120, 190, 255)); y += 34
    for line in state_text.split("\n"):
        for chunk in [line[i : i + 64] for i in range(0, len(line), 64)] or [""]:
            draw.text((x0 + pad, y), chunk, font=f_s, fill=(225, 225, 225)); y += 26
    y += 24
    draw.text((x0 + pad, y), hud["choice"], font=f_m, fill=(120, 190, 255)); y += 36
    bar_x, bar_w = x0 + 220, w - 320
    for name, p in probs:
        col = (86, 214, 120) if name == chosen else (70, 70, 90)
        draw.text((x0 + pad, y + 2), name, font=f_s, fill=(255, 255, 255) if name == chosen else (170, 170, 185))
        draw.rectangle([bar_x, y, bar_x + bar_w, y + 22], fill=(38, 38, 48))
        draw.rectangle([bar_x, y, bar_x + int(bar_w * p), y + 22], fill=col)
        draw.text((bar_x + bar_w + 10, y + 1), f"{100*p:3.0f}%", font=f_s, fill=(220, 220, 220)); y += 32
    y += 18
    draw.text((x0 + pad, y), hud["noul"], font=f_m, fill=(120, 190, 255)); y += 36
    tcol = (230, 80, 80) if threat > 0.5 else (90, 90, 110)
    draw.rectangle([bar_x, y, bar_x + bar_w, y + 22], fill=(38, 38, 48))
    draw.rectangle([bar_x, y, bar_x + int(bar_w * threat), y + 22], fill=tcol)
    draw.text((x0 + pad, y + 2), hud["pyes"], font=f_s, fill=(200, 200, 210))
    draw.text((bar_x + bar_w + 10, y + 1), f"{100*threat:3.0f}%", font=f_s, fill=(220, 220, 220)); y += 56
    draw.text((x0 + pad, y), hud["lat"].format(lat=latency_ms, dps=dps, step=step), font=f_m, fill=(255, 210, 90)); y += 38
    draw.text((x0 + pad, y), hud["stats"].format(hp=hp, kills=kills), font=f_m, fill=(200, 200, 210))
    draw.text((x0 + pad, h - 44), hud["foot"], font=f_s, fill=(120, 120, 140))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="release/OpenThai-SystemOne-v0.2")
    ap.add_argument("--out", default="runs/demo/doom.mp4")
    ap.add_argument("--scenario", default="deadly_corridor")
    ap.add_argument("--seconds", type=float, default=75)
    ap.add_argument("--skill", type=int, default=2)
    ap.add_argument("--tics", type=int, default=4, help="game tics per decision (35 tics = 1 s)")
    ap.add_argument("--episodes", type=int, default=99)
    ap.add_argument("--snapshot", default="", help="also write this PNG of one composed frame")
    ap.add_argument("--lang", default="en", choices=["en", "th"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    hud = HUD[args.lang]
    goal, situation = SCENARIOS.get(args.scenario, SCENARIOS["deadly_corridor"])
    instructions = situation + " " + POLICY.format(goal=goal) + " Pick the single best next action."

    client = SystemOneClient(args.model)
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / f"{args.scenario}.cfg"))
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_objects_info_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_doom_skill(args.skill)
    game.set_seed(args.seed)
    game.init()
    buttons = game.get_available_buttons()
    btn_index = {b: i for i, b in enumerate(buttons)}
    actions = {n: (desc, b) for n, (desc, b) in ACTIONS.items() if b in btn_index}
    criteria = {n: d for n, (d, _) in actions.items()}

    W, H, PW = 1280, 1080, 640  # game view 640x480 -> 1280x960 (2x, crisp) letterboxed in 1280x1080; panel 640 -> 1920x1080
    FD = Path(__file__).resolve().parent / "fonts"
    reg = str(FD / "Sarabun-Regular.ttf") if (FD / "Sarabun-Regular.ttf").exists() else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = str(FD / "Sarabun-Bold.ttf") if (FD / "Sarabun-Bold.ttf").exists() else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fonts = (ImageFont.truetype(bold, 34), ImageFont.truetype(bold, 22), ImageFont.truetype(reg, 19))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fps = 35 // args.tics * args.tics  # we emit one frame per tic -> 35 fps
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W+PW}x{H}", "-r", "35",
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", args.out], stdin=subprocess.PIPE)
    total_frames = int(args.seconds * 35)
    frames = 0; step = 0; lat_hist = deque(maxlen=20); t_start = time.time()
    history = deque(maxlen=4)
    snapshot_done = False
    for ep in range(args.episodes):
        if frames >= total_frames:
            break
        game.new_episode()
        last_pos = None; stuck = False
        while not game.is_episode_finished() and frames < total_frames:
            state = game.get_state()
            player = next((o for o in state.objects if o.name == "DoomPlayer"), None)
            pos = (round(player.position_x), round(player.position_y)) if player else None
            stuck = pos is not None and pos == last_pos and history and history[-1].startswith("MOVE")
            last_pos = pos
            text, enemies = describe_state(game, state, list(history), stuck)
            t0 = time.perf_counter()
            resp = client.system_one({"game": "Doom", "scenario": args.scenario, "state": text}, {"action": Choice(instructions=instructions, criteria=criteria), "threat": Noul(instructions=THREAT_Q)})
            latency = (time.perf_counter() - t0) * 1000
            lat_hist.append(latency)
            a = resp.answers["action"]; chosen = a.choice; threat = resp.answers["threat"].noul
            probs = sorted(a.probabilities.items(), key=lambda kv: -kv[1])
            history.append(chosen)
            step += 1
            dps = 1000 / (sum(lat_hist) / len(lat_hist))
            vec = [0] * len(buttons); vec[btn_index[actions[chosen][1]]] = 1
            game.set_action(vec)
            hp = game.get_game_variable(vzd.GameVariable.HEALTH); kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            for _ in range(args.tics):
                if game.is_episode_finished():
                    break
                game.advance_action(1)
                s2 = game.get_state()
                screen = s2.screen_buffer if s2 is not None and s2.screen_buffer is not None else state.screen_buffer
                canvas = Image.new("RGB", (W + PW, H), (0, 0, 0))
                canvas.paste(Image.fromarray(screen).resize((1280, 960), Image.NEAREST), (0, 60))
                draw = ImageDraw.Draw(canvas)
                # crosshair + chosen action tag on the game view
                cx, cy = W // 2, 60 + 480
                draw.line([(cx - 18, cy), (cx + 18, cy)], fill=(255, 255, 255), width=3)
                draw.line([(cx, cy - 18), (cx, cy + 18)], fill=(255, 255, 255), width=3)
                draw.rectangle([0, 0, W, 60], fill=(10, 10, 14)); draw.rectangle([0, 1020, W, 1080], fill=(10, 10, 14))
                draw.text((24, 14), f"> {chosen}", font=fonts[1], fill=(86, 214, 120))
                draw.text((24, 1034), f"ViZDoom · {args.scenario} · skill {args.skill}", font=fonts[2], fill=(150, 150, 165))
                draw_panel(draw, fonts, hud, W, PW, H, text, probs, chosen, threat, latency, dps, step, hp, kills)
                if args.snapshot and not snapshot_done and step == 12:
                    canvas.save(args.snapshot); snapshot_done = True
                ff.stdin.write(canvas.tobytes()); frames += 1
        print(f"episode {ep+1}: kills={game.get_game_variable(vzd.GameVariable.KILLCOUNT):.0f} health={game.get_game_variable(vzd.GameVariable.HEALTH):.0f} steps={step} frames={frames}", flush=True)
    ff.stdin.close(); ff.wait(); game.close()
    print(f"done: {frames} frames ({frames/35:.0f} s), {step} decisions, mean latency {sum(lat_hist)/max(1,len(lat_hist)):.1f} ms, wall {time.time()-t_start:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
