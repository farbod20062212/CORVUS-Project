# پوشه‌ی دیتا

دیتای آموزشی را اینجا بگذار (یا هر جای دیگر؛ مهم این است که مسیرش را به برنامه بدهی).

| دیتا | کجا | فرمت | دستور |
|---|---|---|---|
| چت | `data/my_chat.csv` | CSV، ستون اول سؤال، ستون دوم جواب، سطر اول هدر (`prompt,response`) | `train-lm --chat data/my_chat.csv` |
| متن خام | فایل‌های `.txt` داخل `data/corpus/` | هر `.txt` یک سند؛ زیرپوشه‌ها هم خوانده می‌شوند | `train-lm --text data/corpus` |
| دسته‌بندی متن | `data/comments.csv` | هدر `text,label` | `train-text --data data/comments.csv` |
| تصویر | `data/images/<نام_کلاس>/*.png` | هر کلاس یک پوشه | `train-image --data data/images` |
| SQL | `data/dump.sql` یا `.sqlite` | اول به CSV تبدیل کن: `python3 tools/sql_to_csv.py --sql data/dump.sql --list` | بعد دستور مربوط به CSV |

نکته‌ها:
- **همه‌ی فایل‌ها UTF-8 باشند** (در Notepad: Save As ← Encoding: UTF-8). BOM و line-endingهای ویندوز (CRLF) خودکار پاک می‌شوند.
- در پوشه‌ی `data/corpus/` فقط فایل‌های آموزشی `.txt` بگذار؛ **هر `.txt` آنجا خوانده می‌شود** (به همین خاطر این راهنما `.md` است).
- مسیر نسبی از **working directory** شروع می‌شود، نه از جایی که فایل اجرایی هست. اگر برنامه فایل را پیدا نکرد، پیام خطا working directory را چاپ می‌کند. مطمئن‌ترین راه: مسیر کامل، مثلاً `C:/data/corpus` (اسلش معمولی در ویندوز هم کار می‌کند).
- جداکننده‌ی CSV می‌تواند `,` یا `;` یا Tab باشد (خودکار). ستون label نباید خالی باشد.
- `sample_chat.csv` فقط برای امتحان است (چند جمله‌ی تکراری).
