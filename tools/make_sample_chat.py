#!/usr/bin/env python3
"""Creates sample_chat.csv (prompt,response) with small Persian + English dialogues to test train-lm."""
import csv, random
random.seed(1)
names = ["علی", "سارا", "رضا", "مینا", "Ali", "Sara"]
qa = [
  ("سلام", ["سلام! چطور می‌تونم کمکت کنم؟", "سلام دوست من، خوش اومدی!"]),
  ("حالت چطوره؟", ["من خوبم، ممنون که پرسیدی. تو چطوری؟"]),
  ("اسمت چیه؟", ["من یک هوش مصنوعی کوچک هستم که با زبان C++ ساخته شدم."]),
  ("لینوکس چیه؟", ["لینوکس یک سیستم‌عامل متن‌باز است که هسته‌اش را لینوس توروالدز نوشته است."]),
  ("هسته چیست؟", ["هسته یا کرنل قلب سیستم‌عامل است و منابع سخت‌افزار را مدیریت می‌کند."]),
  ("پروسه init چیه؟", ["init اولین پروسه‌ای است که کرنل اجرا می‌کند و شناسه‌اش ۱ است."]),
  ("hello", ["Hello! How can I help you today?", "Hi there!"]),
  ("what is linux", ["Linux is an open source operating system kernel created by Linus Torvalds."]),
  ("what is a kernel", ["The kernel is the core of an operating system that manages hardware resources."]),
  ("who are you", ["I am a tiny language model written from scratch in C++."]),
  ("thanks", ["You're welcome!", "Anytime, happy to help."]),
  ("ممنون", ["خواهش می‌کنم!", "قابلی نداشت."]),
]
rows = []
for _ in range(1500):
    q, answers = random.choice(qa)
    rows.append((q, random.choice(answers)))
with open("sample_chat.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["prompt", "response"]); w.writerows(rows)
print("wrote sample_chat.csv with", len(rows), "pairs")
