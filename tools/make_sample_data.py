#!/usr/bin/env python3
"""Generates small synthetic datasets so you can try minnn immediately.
   Requires Pillow (pip install pillow). Creates ./sample_images/<class>/*.png and ./sample_text.csv"""
import random, os, csv
from PIL import Image, ImageDraw
random.seed(3)
for cls in ["circle", "square", "cross"]:
    os.makedirs(f"sample_images/{cls}", exist_ok=True)
    for i in range(250):
        im = Image.new("L", (32, 32), random.randint(0, 40)); d = ImageDraw.Draw(im)
        cx, cy, r, v = random.randint(9, 22), random.randint(9, 22), random.randint(5, 8), random.randint(180, 255)
        if cls == "circle": d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=v, width=2)
        elif cls == "square": d.rectangle([cx-r, cy-r, cx+r, cy+r], outline=v, width=2)
        else:
            d.line([cx-r, cy, cx+r, cy], fill=v, width=2); d.line([cx, cy-r, cx, cy+r], fill=v, width=2)
        im.save(f"sample_images/{cls}/{i}.png")

topics = {
  "sport": "match team goal player coach league score win season stadium".split(),
  "tech": "software code server network cpu kernel linux compiler memory cache".split(),
  "food": "pizza recipe cook flavor spicy dinner sauce bake kitchen taste".split(),
  "ورزش": "بازی تیم گل بازیکن مربی لیگ فوتبال قهرمان ورزشگاه فصل".split(),
  "فناوری": "نرم‌افزار برنامه سرور شبکه پردازنده هسته لینوکس کامپایلر حافظه کد".split(),
  "غذا": "پیتزا دستور آشپزی طعم تند شام سس پخت آشپزخانه مزه".split(),
}
common = "the a of and is very today و در به این که با".split()
rows = []
for lab, ws in topics.items():
    for _ in range(150):
        toks = [random.choice(ws) for _ in range(5)] + [random.choice(common) for _ in range(4)]
        random.shuffle(toks); rows.append((" ".join(toks) + random.choice([".", "،", "؟", "!"]), lab))
random.shuffle(rows)
with open("sample_text.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["text", "label"]); w.writerows(rows)
print("done: sample_images/ and sample_text.csv")
