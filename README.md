# CORVUS — شبکه عصبی و مدل زبانی از صفر با C++17

بدون هیچ کتابخانه‌ی سنگینی: فقط C++17 + `stb_image.h` (داخل پروژه) و اختیاری OpenMP.
پوشش: داده‌ی جدولی، تصویر (CNN)، دسته‌بندی متن، و **مدل زبانی GPT** که می‌تواند چت کند و متن تولید کند.

## کامپایل
**با CMake (پیشنهادی، برای IDE):** پوشه‌ی `minnn` (همان که `CMakeLists.txt` داخلش است) را باز کن.
```bash
cmake -S . -B build            # Release به‌صورت پیش‌فرض
cmake --build build -j
ctest --test-dir build         # همان minnn selftest
./build/minnn selftest
```
- `-DCMAKE_BUILD_TYPE=Debug` برای دیباگ؛ `-DMINNN_NATIVE=OFF` اگر فایل اجرایی را روی CPU دیگری می‌بری.
- اگر `include/third_party/stb_image.h` نبود، خود CMake دانلودش می‌کند (`-DMINNN_FETCH_STB=OFF` برای خاموش‌کردن).
- اگر فایلی از پروژه کم باشد، CMake اسم همه‌ی فایل‌های گم‌شده را می‌گوید. **کل پوشه را کپی کن** (یا از `minnn.zip`)، نه تک‌فایل.
- با GCC 13 و 14 و clang در حالت Release و Debug تست شده. MSVC تست نشده.

**بدون CMake:**
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -Iinclude src/main.cpp src/stb_impl.cpp -o minnn
./minnn selftest        # همه‌ی چک‌ها باید PASS شوند (الان 56 مورد)
```

## اجرا و شروع آموزش (Visual Studio)
1. `File ← Open ← Folder` و پوشه‌ی `minnn` (همان که `CMakeLists.txt` داخلش است) را باز کن.
2. بالای صفحه، کنار نام کانفیگ (`x64-Debug`) را باز کنید ← `Manage Configurations…` ← `+` ← **`x64-Release`**، و آن را انتخاب کن. **Debug حدود ۱۱ برابر کندتر است** (اندازه‌گیری: ۱۰۶۴ در برابر ۱۱۸۷۹ توکن/ثانیه). برای آموزش همیشه Release.
3. `Build ← Build All`. فایل اجرایی: `out/build/x64-Release/minnn.exe` (مسیر دقیق در پنجره‌ی Output نوشته می‌شود).
4. برنامه بدون آرگومان فقط راهنما را چاپ می‌کند. آرگومان دادن، یکی از این دو راه:
   - **ترمینال (ساده‌ترین):** `View ← Terminal`، سپس مثلاً `out\build\x64-Release\minnn.exe train-lm --text data/corpus --preset small --out data/lm.bin` از پوشه‌ی پروژه.
   - **داخل IDE:** از `Debug ← Debug and Launch Settings for minnn` فایل `launch.vs.json` را بساز و `"args"` و `"currentDir": "${workspaceRoot}"` را بگذارید (نمونه: `tools/launch.vs.json.example`). برای اجرا بدون دیباگر و باز ماندن کنسول: `Ctrl+F5`.
- امتحان سریع: `minnn.exe selftest` (باید 47/47) و `minnn.exe demo`.
- کنسول ویندوز را روی UTF-8 می‌گذارم، ولی کنسول قدیمی حروف فارسی را جدا/برعکس نشان می‌دهد (فقط نمایش، خروجی درست است). **Windows Terminal** بهتر است.
- تست‌شده: نسخه‌ی ویندوز با MinGW کامپایل شد (فقط build، بدون اجرا)؛ داخل خود Visual Studio/MSVC تست نشده.

## فایل CSV و دیتای SQL
**CSV — هر دستور چه می‌خواهد** (ستون را هم با شماره می‌شود داد، هم با **نام ستون** در هدر):

| دستور | ستون‌ها | گزینه‌ها |
|---|---|---|
| `train` (جدولی) | ستون‌های **عددی** + یک ستون label | `--label-col` (پیش‌فرض آخرین)، `--task classification\|regression` |
| `train-text` | یک ستون متن + یک ستون label | `--text-col`، `--label-col`، `--header 0` اگر هدر ندارد |
| `train-lm --chat` | سؤال + جواب | `--prompt-col`، `--response-col`، `--header 0` |
| `train-lm --text f.csv` | فقط یک ستون متن | `--text-col comment` |

- جداکننده‌ی ستون‌ها **خودکار** تشخیص داده می‌شود: `,` یا `;` (Excel در بعضی تنظیمات منطقه‌ای) یا Tab. جداکننده‌ی دیگر (مثل `|`) تشخیص داده نمی‌شود و خطای واضح می‌دهد، نه اینکه بی‌صدا روی دیتای خراب آموزش ببیند.
- UTF-8، BOM و CRLF ویندوز مشکلی ندارند. اعداد اعشاری باید با نقطه باشند (`1.5`، نه `1,5`)؛ برای ویرگول اعشاری پیام راهنما می‌دهد.
- **سلول خالی:** در ستون عددی خطا با شماره‌ی سطر و ستون می‌دهد؛ در ستون label **هشدار** می‌دهد (سطرهای خالی یک کلاس جدا می‌شوند). بهتر است همان اول فیلتر کنی.

**SQL — خود minnn فایل SQL نمی‌خواند؛ اول به CSV تبدیلش کن** با `tools/sql_to_csv.py` (Python 3.8+، بدون نصب هیچ پکیجی):
```bash
python3 tools/sql_to_csv.py --sql dump.sql --list                      # جدول‌ها و ستون‌ها و تعداد سطرها
python3 tools/sql_to_csv.py --sql dump.sql --table comments --out data/comments.csv
python3 tools/sql_to_csv.py --db  site.sqlite --query "SELECT text, label FROM comments WHERE label IS NOT NULL AND label != ''" --out data/c.csv
python3 tools/sql_to_csv.py --sql chat.sql --query "SELECT question AS prompt, answer AS response FROM chat" --out data/chat.csv
```
- ورودی: فایل SQLite (`--db`)، dump برنامه‌ی SQLite یا dump مدل **MySQL / MariaDB** (mysqldump، phpMyAdmin) با `--sql`. dump مدل **PostgreSQL** خوانده نمی‌شود (پیام می‌دهد: با `psql \copy` از خود دیتابیس خروجی CSV بگیر).
- dump قدیمی فارسی اگر UTF-8 نبود: `--encoding cp1256`. NULL و BLOB به سلول خالی تبدیل می‌شوند، پس در `--query` فیلتر کن (`WHERE ... IS NOT NULL`).
- اندازه‌گیری: dump ‏48MB (۲۰۰ هزار سطر) در ۷ ثانیه تبدیل شد و حدود **۶ برابر حجم dump** رم گرفت (برآورد: dump ۱GB حدود ۶GB رم می‌خواهد).
- مسیر کامل: SQL ← CSV ← `train-text --data data/c.csv --text-col text --label-col label` (روی دیتای نمونه‌ام تا ۱۰۰٪ دقت رفت؛ دیتای واقعی دقت دیگری دارد).
- تست: `python3 tools/test_sql_to_csv.py` (۱۱ تست) و یک آزمون تصادفی جداگانه: ۵۰۰ سطر متن خصمانه (نقل‌قول، بک‌اسلش، خط جدید، ایموجی، کلمات SQL) در dump مدل MySQL، ۲۰۰۰ سلول، بدون حتی یک اختلاف.

## دیتای آموزشی را کجا بگذارم
هر جا که بخواهی؛ مسیر را با `--text` / `--chat` / `--data` می‌دهی. پیشنهاد: پوشه‌ی **`data/`** کنار پروژه (توضیح کامل و جدول فرمت‌ها: `data/README.md`).
- متن خام: فایل‌های `.txt` در `data/corpus/` ← `train-lm --text data/corpus`
- چت: `data/my_chat.csv` با ستون‌های `prompt,response` ← `train-lm --chat data/my_chat.csv`
- مسیر نسبی از **working directory** حساب می‌شود (در Visual Studio پیش‌فرض پوشه‌ی فایل اجرایی است، نه پروژه) → از `currentDir` یا مسیر کامل استفاده کن. اگر فایل پیدا نشود، خطا working directory را چاپ می‌کند.
- فایل‌ها UTF-8 باشند؛ BOM و CRLF ویندوز خودکار پاک می‌شوند.

## ساختار
| فایل | کار |
|---|---|
| `tensor.hpp` | Matrix + ضرب ماتریس (register-blocked، OpenMP) |
| `layers.hpp` | Dense، Activation، Dropout، Conv2D، MaxPool2D، Embedding، BatchNorm2D، GlobalAvgPool، ResidualBlock |
| `loss.hpp` / `optimizer.hpp` | CrossEntropy، MSE / SGD، Adam(W) با ذخیره‌ی state |
| `dataset.hpp` `text.hpp` | CSV، توکنایزر فارسی‌پسند |
| `imageproc.hpp` | کلاس `Image` (resize ضدالایاسینگ، fit)، ترنسفورم‌های augmentation، `ImageAugmenter` |
| `image.hpp` | خواندن تصویر؛ کلاس‌های `ImageSource` (`FolderImageSource`، `IdxImageSource`) |
| `model.hpp` | `ModelBuilder` (MLP / CNN / Text)، `Model`، ذخیره/بارگذاری |
| `trainer.hpp` | `Trainer` + `TrainCallback` (`ProgressLogger`، `EarlyStopping`، `CheckpointSaver`) |
| `command.hpp` | `Command` / `CommandRegistry` (همه‌ی دستورهای CLI) |
| `bpe.hpp` | توکنایزر BPE سطح-بایت (هر متن UTF-8) |
| `transformer.hpp` | GPT با backward دستی، KV-cache، `growLM`، `estimateLM` |
| `lm.hpp` | `train-lm` / `chat` / `generate` / `info` / `grow` |
| `selftest.hpp` | تست خودکار همه‌ی اجزا |

## مدل زبانی (چت)
دیتا: فایل CSV با ستون‌های `prompt,response` و/یا متن خام (`.txt` یا پوشه).
```bash
python3 tools/make_sample_chat.py                       # دیتای نمونه
./minnn train-lm --chat sample_chat.csv --preset tiny --steps 900 --lr 2e-3 --out lm.bin
./minnn chat --model lm.bin                             # گفتگو (/reset ، /quit)
./minnn generate --model lm.bin --prompt "سلام" --max-new 100
./minnn train-lm --text corpus/ --chat pairs.csv --preset small   # pretrain + chat با هم
```
- `Ctrl+C` وسط آموزش: ذخیره‌ی `lm.bin.last` + `lm.bin.last.opt` و ادامه با `--resume lm.bin.last`
- `--resume` وزن‌ها و state بهینه‌ساز (`.opt`) را برمی‌گرداند
- `--accum K`: batch مؤثر = batch×K بدون مصرف رام بیشتر
- نمونه‌گیری: `--temp 0.8 --top-k 40 --top-p 0.95 --rep-penalty 1.0` (برای تکرار زیاد 1.1 امتحان کن)

## «تبدیل به large»: چه چیزی واقعاً لازم است
اندازه‌ی مدل با **دیتا و محاسبه** تعیین می‌شود، نه با یک سوییچ. این پروژه سه ابزار می‌دهد:

1. **`--preset tiny|small|medium|large|xl`** — اندازه‌ی مدل، vocab، batch و lr را یک‌جا تنظیم می‌کند (هر فلگ صریح، پریست را override می‌کند).
2. **`./minnn info --preset large`** — قبل از شروع، پارامتر، رام و هزینه را حساب می‌کند.
3. **`./minnn grow --model lm.bin --layers 12 --ctx 512`** — مدل آموزش‌دیده را بزرگ‌تر می‌کند بدون از دست دادن آنچه یاد گرفته (لایه‌های جدید identity هستند؛ با تست ثابت شد). بعدش `train-lm --resume` بزن.
   - فقط عمق و context رشد می‌کند، نه عرض (`dim`).
   - رشد **لایه**: loss تغییر نمی‌کند (اندازه‌گیری: 0.028 → 0.029). رشد **ctx**: جایگاه‌های جدید آموزش‌ندیده‌اند و loss موقتاً بالا می‌رود تا دوباره آموزش ببینند.

برآوردها از خود `info` (اعداد RAM و FLOP)؛ ستون «۸ GFLOPS» از سرعت واقعاً اندازه‌گیری‌شده‌ی من روی ۱ هسته است (~۸۹۰۰ توکن/ثانیه برای tiny). ستون «۵۰ GFLOPS» **برآورد** من برای ۸ ترد است و اندازه‌گیری نشده — سرعت واقعی خودت را از خط `tok/s` همان اول آموزش بخوان.

| preset | پارامتر | RAM آموزش | توکن لازم (~۲۰×پارامتر) | زمان با 8 GFLOPS (اندازه‌گیری‌شده، ۱ هسته) | زمان با 50 GFLOPS (برآورد، ۸ ترد) |
|---|---|---|---|---|---|
| `tiny` | 0.14M | 0.02 GB | 3M | 6 دقیقه | 1 دقیقه |
| `small` | 0.97M | 0.13 GB | 19M | 4.1 ساعت | 39 دقیقه |
| `medium` | 5.64M | 0.85 GB | 113M | 6 روز | 22.7 ساعت |
| `large` | 28.69M | 2.4 GB | 574M | 0.4 سال | 25 روز |
| `xl` | 92.19M | 6.9 GB | 1844M | 4.5 سال | 0.7 سال |

**واقع‌بینی:**
- تا `small` روی CPU همین سیستم عملی است؛ `medium` چند ده ساعت؛ `large` و `xl` با CPU عملاً نشدنی‌اند (و RX 580 هنوز استفاده نمی‌شود؛ بک‌اند GPU ننوشته‌ام).
- مدل بزرگ بدون دیتای کافی بدتر می‌شود نه بهتر (overfit). قاعده: توکن ≈ ۲۰× تعداد پارامتر. اگر دیتای تو چند مگابایت است، `tiny`/`small` درست است.
- مدلی که از صفر روی دیتای شخصی آموزش ببیند الگوهای همان دیتا را تقلید می‌کند؛ «فهم عمومی» مدل‌های بزرگ از میلیاردها توکن می‌آید. برای کیفیت شبیه ChatGPT باید یک مدل متن‌باز از پیش‌آموزش‌دیده را fine-tune کنی، نه این پروژه را بزرگ کنی.

## پردازش تصویر
```bash
./minnn train-image --data data/ --size 64 --channels 3 --fit crop --conv 16,32,64 --augment --epochs 40
./minnn predict --model model.bin --data photo.jpg --top 3 --tta     # tta: میانگین تصویر اصلی و آینه‌ایش
```
ساختار پوشه: `data/<کلاس>/*.png|jpg|bmp|...` (یا فایل‌های MNIST با `--idx-images/--idx-labels`).
- **resize درست:** کوچک‌کردن با میانگین‌گیری سطحی (ضدالایاسینگ)، بزرگ‌کردن bilinear. `--fit stretch|crop|pad` برای نسبت ابعاد متفاوت.
- **نرمال‌سازی** per-channel، فقط از دیتای آموزش محاسبه می‌شود.
- **augmentation** روی پیکسل‌های خام (قبل از نرمال‌سازی): `--augment` یا تک‌تک `--rotate DEG --zoom 0.9,1.1 --shift 0.1 --hflip --vflip --jitter 0.2 --noise 0.02 --cutout 0.3`.
- **معماری:** `--arch plain|resnet`، `--bn`، `--gap`، `--blocks N`.

### نتیجه‌ی اندازه‌گیری‌شده (واقعی، نه ادعا)
بنچمارک کوچک و عمداً سخت: آموزش روی ۲۰۰ تصویر تمیز، تست روی ۳۲۰ تصویر «کثیف» (چرخش تا ±۳۰°، جابه‌جایی، تغییر اندازه، نویز). ۴ کلاس، ۶۰ epoch، batch 16، میانگین ۳ seed. بازتولید: `python3 tools/make_bench_images.py` و `tools/score_predictions.py`.

| مدل | دقت روی تست کثیف |
|---|---|
| plain، بدون augmentation | 76.9% |
| plain + `--augment` | 85.7% |
| plain + `--bn` + `--augment` | 86.1% |
| ResNet دو مرحله‌ای + `--augment` | 57.2% |

- **augmentation حدود ۹ امتیاز اضافه کرد** و این واضح‌ترین برد است.
- **BatchNorm روی این شبکه‌ی کوچک فرقی نکرد** (اختلاف داخل نویز؛ سرعت همگرایی هم یکسان بود). روی شبکه‌های عمیق‌تر امتحان نکرده‌ام.
- **ResNet کم‌عمق در هر ۳ seed بدتر بود** (دقت آموزش ۱۰۰٪، تست ۵۷٪). با عمق بیشتر بهتر شد (`--conv 8,16,32 --blocks 2`: ۷۸.۷٪ روی یک seed) ولی باز کمتر از plain. علتش را حدس می‌زنم میدان دید کوچک باشد؛ ثابت‌شده نیست. نتیجه: روی دیتای کم، ResNet ارتقای رایگان نیست.
- محدودیت: این یک بنچمارک ساختگی و کوچک است؛ روی عکس‌های واقعی ممکن است نتیجه فرق کند.

## متن / جدولی
```bash
./minnn train-text  --data comments.csv --ngrams 2
./minnn train       --data data.csv --hidden 128,64
./minnn predict     --model model.bin --data ...     # برای هر نوع مدل
./minnn info        --model model.bin
```
جزئیات هر گزینه: `./minnn` بدون آرگومان. `--checkpoint` بهترین مدل را بعد از هر بهبود روی دیسک می‌نویسد.

## طراحی شی‌گرا
| مفهوم | کلاس‌ها | فایده |
|---|---|---|
| لایه‌ها | `Layer` ← `Dense`، `Conv2D`، `BatchNorm2D`، `ResidualBlock` (یک Layer که Layerهای دیگر را در خود دارد) | هر لایه‌ی جدید فقط `forward/backward` می‌خواهد |
| ساخت مدل | `ModelBuilder` ← `MLPBuilder`، `CNNBuilder`، `TextBuilder` | نوع مدل جدید = یک builder جدید |
| ترنسفورم تصویر | `ImageTransform` ← `RandomAffine`، `ColorJitter`، `Cutout`، ... و `Compose` | augmentation قابل ترکیب |
| منبع داده | `ImageSource` ← `FolderImageSource`، `IdxImageSource` | منبع جدید (مثلاً CSV مسیرها) بدون تغییر کد آموزش |
| آموزش | `Trainer` + `TrainCallback` ← `ProgressLogger`، `EarlyStopping`، `CheckpointSaver` | رفتار جدید (مثلاً لاگ در فایل) = یک callback، بدون دست‌زدن به حلقه |
| CLI | `Command` ← `TrainImageCommand`، `FnCommand`؛ `CommandRegistry` | افزودن دستور = یک خط ثبت |

صادقانه: `TrainImageCommand` یک کلاس واقعی با متدهای جدا برای هر مرحله است؛ بقیه‌ی دستورها فعلاً فقط با `FnCommand` به‌صورت adapter دور توابع قدیمی پیچیده شده‌اند. بازنویسی آموزش LM (`train-lm`) به `Trainer` هنوز انجام نشده؛ حلقه‌ی خودش را دارد.

## تست
`./minnn selftest` این‌ها را می‌سنجد: ضرب ماتریس در برابر مرجع دقیق، backprop همه‌ی لایه‌ها با تفاضل متناهی (Conv و Pool جداگانه و دقیق)، backprop کامل GPT، round-trip توکنایزر، KV-cache در برابر forward آموزش، ذخیره/بارگذاری، grow، ادامه‌ی Adam، پارسر CSV و توکنایزر فارسی، و اینکه GPT واقعاً loss را پایین می‌آورد.
تست‌ها با **mutation testing** سنجیده شده‌اند: با خراب‌کردن عمدی backward (attention، LayerNorm، GELU، Conv، Pool، head) باید fail شوند — هر ۸ مورد fail شد.

**تست رفتار شی‌گرا:** `Trainer` با callback توقف، `EarlyStopping` (شمارنده، ریست، بازگردانی بهترین وزن‌ها)، `CheckpointSaver`، `CommandRegistry` و `ImageSource` هر کدام تست دارند و با mutation testing سنجیده شده‌اند (۶ خرابی عمدی، هر ۶ گرفته شد). رفتار آموزش بعد از بازنویسی با همان seed بایت‌به‌بایت برابر مدل قبل از بازنویسی است.
