#!/usr/bin/env python3
"""sql_to_csv.py - turn SQL data into the UTF-8 CSV files that minnn reads.  Python 3.8+, standard library only.

  python3 tools/sql_to_csv.py --sql dump.sql --list                      what tables / columns are inside
  python3 tools/sql_to_csv.py --sql dump.sql --table comments --out data/comments.csv
  python3 tools/sql_to_csv.py --sql dump.sql --query "SELECT text, label FROM comments WHERE label != ''" --out data/c.csv
  python3 tools/sql_to_csv.py --db  site.sqlite --table posts --out data/posts.csv

Input kinds
  --db   FILE   a SQLite database file (.db .sqlite .sqlite3)
  --sql  FILE   a SQL dump: SQLite (.dump) or MySQL / MariaDB (mysqldump, phpMyAdmin).  The MySQL dump is converted to SQLite
                on the fly (best effort: typical dumps work; exotic features are skipped with a warning).
  PostgreSQL dumps are not read: export from the database itself with  psql -c "\\copy (SELECT ...) TO 'out.csv' CSV HEADER"
Old Persian dumps are often not UTF-8: add  --encoding cp1256  (or utf-16).
"""
import argparse
import csv
import re
import sqlite3
import sys

PH_OPEN, PH_CLOSE = "\ue000", "\ue001"      # private-use characters that mark "string literal number N"


# ----------------------------------------------------------------------------- dialect detection
def detect_dialect(text):
    if re.search(r"PostgreSQL database dump|^COPY\s+\S+.*\sFROM\s+stdin;", text, re.M):
        return "postgres"
    head = text[:300000]
    if re.search(r"ENGINE\s*=|/\*!\d{5}|LOCK TABLES|mysqldump|MariaDB dump|AUTO_INCREMENT|ON UPDATE CURRENT_TIMESTAMP|`", head):
        return "mysql"
    return "sqlite"


# ----------------------------------------------------------------------------- MySQL -> SQLite
_ESC = {"n": "\n", "r": "\r", "t": "\t", "0": "", "b": "\b", "Z": "\x1a"}


def _lift_strings(sql):
    """Replace every '...' literal by a placeholder (decoding MySQL backslash escapes), drop comments, convert `ident`
    to "ident".  Returns (skeleton, strings).  After this step nothing inside a string can be mistaken for SQL."""
    out, strings, i, n = [], [], 0, len(sql)
    while i < n:
        c = sql[i]
        if c == "'":
            j, buf = i + 1, []
            while True:
                if j >= n:
                    raise ValueError("unterminated string literal starting at character %d" % i)
                ch = sql[j]
                if ch == "\\" and j + 1 < n:
                    nx = sql[j + 1]
                    if nx in _ESC:
                        buf.append(_ESC[nx])
                    elif nx in "%_":
                        buf.append("\\" + nx)          # MySQL keeps the backslash for \% and \_
                    else:
                        buf.append(nx)                 # \'  \"  \\  and any other escaped character
                    j += 2
                elif ch == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        buf.append("'"); j += 2
                    else:
                        break
                else:
                    buf.append(ch); j += 1
            strings.append("".join(buf))
            out.append("%s%d%s" % (PH_OPEN, len(strings) - 1, PH_CLOSE))
            i = j + 1
        elif c == "-" and sql.startswith("--", i) and (i + 2 >= n or sql[i + 2] in " \t\r\n"):
            while i < n and sql[i] != "\n":
                i += 1
        elif c == "/" and sql.startswith("/*", i):                   # /* ... */ and MySQL's /*!40101 ... */
            end = sql.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated /* comment")
            out.append(" "); i = end + 2
        elif c == "`":
            j = i + 1; ident = []
            while j < n:
                if sql[j] == "`":
                    if j + 1 < n and sql[j + 1] == "`":
                        ident.append("`"); j += 2; continue
                    break
                ident.append(sql[j]); j += 1
            out.append('"' + "".join(ident).replace('"', '""') + '"')
            i = j + 1
        else:
            out.append(c); i += 1
    return "".join(out), strings


def _split_top(body):
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


_DROP_DEF = re.compile(r"^\s*(UNIQUE\s+)?(KEY|INDEX)\b|^\s*(FULLTEXT|SPATIAL)\b|^\s*CONSTRAINT\b|^\s*FOREIGN\s+KEY\b", re.I)
_COL_JUNK = [
    (re.compile(r"\bunsigned\b|\bzerofill\b|\bAUTO_INCREMENT\b", re.I), ""),
    (re.compile(r"\b(CHARACTER\s+SET|CHARSET)\s+\w+", re.I), ""),
    (re.compile(r"\bCOLLATE\s+\w+", re.I), ""),
    (re.compile(r"\bCOMMENT\s+" + PH_OPEN + r"\d+" + PH_CLOSE, re.I), ""),
    (re.compile(r"\bON\s+UPDATE\s+\w+(\(\d*\))?", re.I), ""),
    (re.compile(r"\b(enum|set)\s*\([^)]*\)", re.I), "TEXT"),
]


def _clean_create(st):
    a, b = st.find("("), st.rfind(")")
    if a < 0 or b < a:
        return st
    defs = []
    for d in _split_top(st[a + 1:b]):
        if _DROP_DEF.match(d):
            continue
        for rx, rep in _COL_JUNK:
            d = rx.sub(rep, d)
        defs.append(" ".join(d.split()))
    return st[:a] + "(" + ", ".join(defs) + ")" if defs else st      # table options after ")" (ENGINE=..., CHARSET=...) dropped


_SKIP_STMT = re.compile(r"^(LOCK|UNLOCK)\s+TABLES|^SET\s|^USE\s|^CREATE\s+(DATABASE|SCHEMA)|^START\s+TRANSACTION|^COMMIT$|^DELIMITER\b"
                        r"|^ALTER\s+TABLE\s+\S+\s+(DISABLE|ENABLE)\s+KEYS|^CREATE\s+(OR\s+REPLACE\s+)?(ALGORITHM|DEFINER|TRIGGER|PROCEDURE|FUNCTION|EVENT|VIEW)", re.I)


def mysql_to_sqlite_statements(sql):
    """MySQL/MariaDB dump text -> list of SQLite statements."""
    skeleton, strings = _lift_strings(sql)
    restore = re.compile(PH_OPEN + r"(\d+)" + PH_CLOSE)
    stmts = []
    for st in skeleton.split(";"):
        st = " ".join(st.split())
        if not st or _SKIP_STMT.match(st):
            continue
        if re.match(r"CREATE\s+TABLE", st, re.I):
            st = _clean_create(st)
        elif re.match(r"INSERT\s+IGNORE", st, re.I):
            st = re.sub(r"^INSERT\s+IGNORE", "INSERT OR IGNORE", st, flags=re.I)
        st = re.sub(r"_(?:utf8mb4|utf8|latin1|binary)\s*(?=" + PH_OPEN + ")", "", st)
        st = re.sub(r"\b_binary\s+", "", st, flags=re.I)
        st = restore.sub(lambda m: "'" + strings[int(m.group(1))].replace("'", "''").replace("\x00", "") + "'", st)
        stmts.append(st)
    return stmts


# ----------------------------------------------------------------------------- loading
def read_text(path, encoding):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8-sig" if encoding.lower() in ("utf-8", "utf8") else encoding)
    except UnicodeDecodeError as e:
        raise SystemExit("cannot decode %s as %s (%s).\n  Try --encoding cp1256 (old Persian files) or --encoding utf-16." % (path, encoding, e))


def load_sql(path, encoding="utf-8", dialect="auto"):
    text = read_text(path, encoding)
    if dialect == "auto":
        dialect = detect_dialect(text)
    if dialect == "postgres":
        raise SystemExit("this looks like a PostgreSQL dump (COPY ... FROM stdin), which this tool does not read.\n"
                         "  Import it with psql, then export a table:  psql -d DB -c \"\\copy (SELECT * FROM t) TO 'out.csv' CSV HEADER\"")
    con = sqlite3.connect(":memory:")
    warnings = []
    if dialect == "sqlite":
        try:
            con.executescript(text)
        except sqlite3.Error as e:
            raise SystemExit("SQLite could not run this dump: %s\n  If it is a MySQL dump, add --dialect mysql." % e)
    else:
        for st in mysql_to_sqlite_statements(text):
            try:
                con.execute(st)
            except sqlite3.Error as e:
                warnings.append((st[:90], str(e)))
        con.commit()
    return con, dialect, warnings


def tables_of(con):
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def q(name):
    return '"' + name.replace('"', '""') + '"'


def export(con, query, out, limit=None):
    cur = con.execute(query + (" LIMIT %d" % limit if limit else ""))
    cols = [d[0] for d in cur.description]
    blobs = rows = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for row in cur:
            cells = []
            for v in row:
                if v is None:
                    cells.append("")
                elif isinstance(v, (bytes, bytearray)):
                    cells.append(""); blobs += 1
                elif isinstance(v, float):
                    cells.append(repr(v))
                else:
                    cells.append(v)
            w.writerow(cells); rows += 1
    return cols, rows, blobs


def main(argv=None):
    ap = argparse.ArgumentParser(description="SQL / SQLite -> CSV for minnn", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="SQLite database file")
    src.add_argument("--sql", help="SQL dump file (SQLite or MySQL/MariaDB)")
    ap.add_argument("--dialect", choices=["auto", "sqlite", "mysql"], default="auto")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--list", action="store_true", help="show tables and columns")
    ap.add_argument("--table", help="export this table (SELECT *)")
    ap.add_argument("--query", help="export the result of this SELECT")
    ap.add_argument("--out", help="output .csv file")
    ap.add_argument("--limit", type=int, help="export at most N rows")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if a.db:
        con, dialect, warnings = sqlite3.connect("file:%s?mode=ro" % a.db, uri=True), "sqlite database", []
    else:
        con, dialect, warnings = load_sql(a.sql, a.encoding, a.dialect)
        print("read %s as %s dump" % (a.sql, dialect))
    if warnings:
        print("warning: %d statement(s) skipped (they do not affect plain tables):" % len(warnings))
        for st, err in warnings[:5]:
            print("  - %s ...  [%s]" % (st, err))
    tables = tables_of(con)
    if not tables:
        raise SystemExit("no tables were found/loaded - nothing to export.")

    if a.list or not (a.table or a.query):
        for t in tables:
            n = con.execute("SELECT COUNT(*) FROM %s" % q(t)).fetchone()[0]
            cols = ", ".join("%s %s" % (r[1], r[2] or "") for r in con.execute("PRAGMA table_info(%s)" % q(t)))
            print("table %s  (%d rows)\n    %s" % (t, n, cols))
        if not (a.table or a.query):
            print("\nexport one with:  --table NAME --out file.csv     or     --query \"SELECT ...\" --out file.csv")
        return 0
    if not a.out:
        raise SystemExit("--out FILE.csv is required with --table / --query")
    if a.table:
        if a.table not in tables:
            raise SystemExit("no table '%s'. Tables: %s" % (a.table, ", ".join(tables)))
        query = "SELECT * FROM %s" % q(a.table)
    else:
        query = a.query.rstrip().rstrip(";")
    try:
        cols, rows, blobs = export(con, query, a.out, a.limit)
    except sqlite3.Error as e:
        raise SystemExit("query failed: %s" % e)
    print("wrote %s: %d rows, columns: %s" % (a.out, rows, ", ".join(cols)))
    if blobs:
        print("note: %d binary (BLOB) cells were written as empty." % blobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
