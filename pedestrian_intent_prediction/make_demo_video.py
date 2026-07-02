"""
make_demo_video.py

Stitch a directory of PNG frames into an mp4 (via cv2.VideoWriter)
and an animated GIF (via imageio if available, else pure-Python PIL fallback).
"""

import argparse
from pathlib import Path

import cv2


def parse_args():
    ap = argparse.ArgumentParser(description="Stitch frames into video/gif")
    ap.add_argument("--frames_dir", type=Path, required=True)
    ap.add_argument("--out_mp4", type=Path, default=None)
    ap.add_argument("--out_gif", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=6)
    return ap.parse_args()


def main():
    args = parse_args()
    frames = sorted(args.frames_dir.glob("*.png"))
    if not frames:
        print(f"[WARN] No frames in {args.frames_dir}")
        return

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    if args.out_mp4 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(args.out_mp4), fourcc, args.fps, (w, h))
        for p in frames:
            img = cv2.imread(str(p))
            if img is not None:
                vw.write(img)
        vw.release()
        print(f"Wrote {args.out_mp4}")

    if args.out_gif is not None:
        # try imageio, else PIL, else give up
        try:
            import imageio.v2 as imageio
            imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frames]
            imageio.mimsave(str(args.out_gif), imgs, fps=args.fps)
            print(f"Wrote {args.out_gif}")
            return
        except ImportError:
            pass
        try:
            from PIL import Image
            pil_imgs = [Image.open(str(p)).convert("P", palette=Image.ADAPTIVE) for p in frames]
            pil_imgs[0].save(
                str(args.out_gif), save_all=True, append_images=pil_imgs[1:],
                duration=int(1000 / args.fps), loop=0, optimize=True,
            )
            print(f"Wrote {args.out_gif}")
        except ImportError:
            print("[WARN] Neither imageio nor PIL available; skipping GIF")


if __name__ == "__main__":
    main()
