#!/usr/bin/env python3
"""Accuracy of `minnn predict --data <folder>` output (saved to a file); the true class is the image folder name.
   Usage: python3 tools/score_predictions.py preds.txt"""
import sys, os, re
ok=n=0
for line in open(sys.argv[1]):
    m=re.match(r"(.+\.png): (\w+) ",line)
    if m: n+=1; ok+= os.path.basename(os.path.dirname(m.group(1)))==m.group(2)
print(f"{ok}/{n} = {ok/max(n,1):.3f}")
