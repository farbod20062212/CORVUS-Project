#!/usr/bin/env python3
"""Benchmark for the image pipeline: bench/train (clean, upright shapes) and bench/test (rotated, shifted, rescaled, noisy).
   Needs Pillow.  Usage:  python3 tools/make_bench_images.py  then train on bench/train and predict on bench/test."""
import random, math, os
from PIL import Image, ImageDraw
def shape(cls, rng, rot_max, shift, zoom, noise):
    S=64; im=Image.new("L",(S,S),rng.randint(0,40)); d=ImageDraw.Draw(im)
    cx=S/2+rng.uniform(-shift,shift); cy=S/2+rng.uniform(-shift,shift)
    r=S*0.28*rng.uniform(1-zoom,1+zoom); th=math.radians(rng.uniform(-rot_max,rot_max)); v=rng.randint(170,255)
    def rot(pts): return [(cx+(x-cx)*math.cos(th)-(y-cy)*math.sin(th), cy+(x-cx)*math.sin(th)+(y-cy)*math.cos(th)) for x,y in pts]
    if cls=="circle": d.ellipse([cx-r,cy-r,cx+r,cy+r],outline=v,width=4)
    elif cls=="square": d.polygon(rot([(cx-r,cy-r),(cx+r,cy-r),(cx+r,cy+r),(cx-r,cy+r)]),outline=v,width=4)
    elif cls=="triangle": d.polygon(rot([(cx,cy-r),(cx+r,cy+r),(cx-r,cy+r)]),outline=v,width=4)
    else:
        for a,b in [((cx-r,cy),(cx+r,cy)),((cx,cy-r),(cx,cy+r))]: d.line(rot([a,b]),fill=v,width=4)
    px=im.load()
    for y in range(S):
        for x in range(S):
            px[x,y]=max(0,min(255,int(px[x,y]+rng.gauss(0,noise))))
    return im
rng=random.Random(5)
for split,n,rot,sh,zm,nz in [("train",50,5,2,0.05,6),("test",80,30,6,0.2,18)]:
    for cls in ["circle","square","triangle","plus"]:
        os.makedirs(f"bench/{split}/{cls}",exist_ok=True)
        for i in range(n): shape(cls,rng,rot,sh,zm,nz).save(f"bench/{split}/{cls}/{i}.png")
print("ok")
