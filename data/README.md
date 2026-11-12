# Data Directory

Place your training data here (or anywhere else; the important thing is to provide its path to the program).

| Data type | Location | Format | Command |
|---|---|---|---|
| Chat | `data/my_chat.csv` | CSV; first column is the question, second column is the answer, first row is the header (`prompt,response`) | `train-lm --chat data/my_chat.csv` |
| Raw text | `.txt` files inside `data/corpus/` | Each `.txt` file is one document; subdirectories are also read | `train-lm --text data/corpus` |
| Text classification | `data/comments.csv` | Header: `text,label` | `train-text --data data/comments.csv` |
| Images | `data/images/<class_name>/*.png` | One directory per class | `train-image --data data/images` |
| SQL | `data/dump.sql` or `.sqlite` | Convert to CSV first: `python3 tools/sql_to_csv.py --sql data/dump.sql --list` | Then use the relevant CSV command |

## Notes

- **All files must be UTF-8 encoded** (in Notepad: Save As → Encoding: UTF-8). BOMs and Windows line endings (CRLF) are stripped automatically.
- Put only `.txt` training files in `data/corpus/`; **every `.txt` file in that directory is read** (this is why this guide uses the `.md` extension).
- Relative paths are resolved from the **working directory**, not from the executable's location. If the program cannot find a file, the error message prints the working directory. The safest option is to use an absolute path, for example `C:/data/corpus` (forward slashes also work on Windows).
- CSV delimiters can be `,`, `;`, or Tab (detected automatically). The label column must not be empty.
- `sample_chat.csv` is only for testing (it contains a few repeated sentences).
