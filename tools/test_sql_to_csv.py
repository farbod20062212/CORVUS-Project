#!/usr/bin/env python3
"""Tests for sql_to_csv.py.   Run:  python3 tools/test_sql_to_csv.py"""
import contextlib
import csv
import io
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sql_to_csv as S  # noqa: E402

# Shaped like real `mysqldump` output.  The row data is deliberately hostile: text that *looks like* SQL.
MYSQL_DUMP = r"""-- MySQL dump 10.13  Distrib 8.0.33, for Linux (x86_64)
--
-- Host: localhost    Database: shop
-- ------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET TIME_ZONE='+00:00' */;

--
-- Table structure for table `comments`
--

DROP TABLE IF EXISTS `comments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
CREATE TABLE `comments` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `author` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'who; wrote it, (really)',
  `body` longtext COLLATE utf8mb4_unicode_ci,
  `sentiment` enum('pos','neg','neu') NOT NULL DEFAULT 'neu',
  `score` decimal(10,2) DEFAULT NULL,
  `created` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_author` (`author`),
  UNIQUE KEY `uq_body` (`body`(191)),
  CONSTRAINT `fk_a` FOREIGN KEY (`author`) REFERENCES `users` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='ENGINE=fake; KEY x';
/*!40101 SET character_set_client = @saved_cs_client */;

LOCK TABLES `comments` WRITE;
/*!40000 ALTER TABLE `comments` DISABLE KEYS */;
INSERT INTO `comments` VALUES (1,'Ali','It\'s great, really; (10/10) -- not a comment /* nor this */','pos',9.50,'2024-01-01 10:00:00'),(2,'سارا','خیلی خوب بود\nخط دوم با "نقل‌قول" و بک‌اسلش \\ و `بک‌تیک`','pos',NULL,'2024-01-02 10:00:00'),(3,NULL,'ENGINE=InnoDB; LOCK TABLES `x` WRITE; DROP TABLE comments;','neg',1.00,'2024-01-03 10:00:00');
INSERT INTO `comments` VALUES (4,'Bob','C:\\temp\\new','neu',0.00,'2024-01-04 00:00:00'),(5,'O''Neil','doubled '''' quotes and a 100\% sign','neg',2.25,'2024-01-05 00:00:00'),(6,'Eve',_utf8mb4'with introducer','pos',3.00,'2024-01-06 00:00:00');
/*!40000 ALTER TABLE `comments` ENABLE KEYS */;
UNLOCK TABLES;

DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `id` int NOT NULL,
  `name` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4;
INSERT INTO `users` VALUES (1,'ali'),(2,'sara');

DELIMITER ;;
/*!50003 CREATE*/ /*!50017 DEFINER=`root`@`localhost`*/ /*!50003 TRIGGER `trg` BEFORE INSERT ON `users` FOR EACH ROW BEGIN SET NEW.name = UPPER(NEW.name); END */;;
DELIMITER ;
-- Dump completed on 2024-02-01 12:00:00
"""

EXPECTED_COMMENTS = [
    ["1", "Ali", "It's great, really; (10/10) -- not a comment /* nor this */", "pos", "9.5", "2024-01-01 10:00:00"],
    ["2", "سارا", 'خیلی خوب بود\nخط دوم با "نقل‌قول" و بک‌اسلش \\ و `بک‌تیک`', "pos", "", "2024-01-02 10:00:00"],
    ["3", "", "ENGINE=InnoDB; LOCK TABLES `x` WRITE; DROP TABLE comments;", "neg", "1.0", "2024-01-03 10:00:00"],
    ["4", "Bob", "C:\\temp\\new", "neu", "0.0", "2024-01-04 00:00:00"],
    ["5", "O'Neil", "doubled '' quotes and a 100\\% sign", "neg", "2.25", "2024-01-05 00:00:00"],
    ["6", "Eve", "with introducer", "pos", "3.0", "2024-01-06 00:00:00"],
]


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = S.main(argv)
    return rc, buf.getvalue()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


class SqlToCsv(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = lambda name: os.path.join(self.dir.name, name)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, text, enc="utf-8"):
        with open(self.p(name), "wb") as f:
            f.write(text.encode(enc))
        return self.p(name)

    # ---- MySQL dump
    def test_mysql_dump_table_roundtrip_with_hostile_text(self):
        dump = self.write("shop.sql", MYSQL_DUMP)
        rc, out = run(["--sql", dump, "--table", "comments", "--out", self.p("c.csv")])
        rows = read_csv(self.p("c.csv"))
        self.assertEqual(rows[0], ["id", "author", "body", "sentiment", "score", "created"])
        got, want = rows[1:], EXPECTED_COMMENTS
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertEqual(g[:4] + g[5:], w[:4] + w[5:])                 # every text/int/date cell identical
            # `score` is decimal(10,2): SQLite's numeric affinity may print 1.00 as "1"; the value must be the same
            self.assertEqual(g[4] == "", w[4] == "")
            if w[4] != "":
                self.assertEqual(float(g[4]), float(w[4]))
        self.assertIn("mysql", out)

    def test_mysql_dump_other_table_and_trigger_ignored(self):
        dump = self.write("shop.sql", MYSQL_DUMP)
        run(["--sql", dump, "--table", "users", "--out", self.p("u.csv")])
        self.assertEqual(read_csv(self.p("u.csv")), [["id", "name"], ["1", "ali"], ["2", "sara"]])

    def test_query_and_limit(self):
        dump = self.write("shop.sql", MYSQL_DUMP)
        run(["--sql", dump, "--query", "SELECT body, sentiment FROM comments WHERE sentiment = 'neg' ORDER BY id;", "--out", self.p("q.csv")])
        rows = read_csv(self.p("q.csv"))
        self.assertEqual(rows[0], ["body", "sentiment"])
        self.assertEqual([r[1] for r in rows[1:]], ["neg", "neg"])
        run(["--sql", dump, "--table", "comments", "--limit", "2", "--out", self.p("l.csv")])
        self.assertEqual(len(read_csv(self.p("l.csv"))), 3)            # header + 2

    def test_list_shows_tables_columns_and_row_counts(self):
        dump = self.write("shop.sql", MYSQL_DUMP)
        _, out = run(["--sql", dump, "--list"])
        self.assertIn("table comments  (6 rows)", out)
        self.assertIn("table users  (2 rows)", out)
        self.assertIn("sentiment", out)

    # ---- SQLite
    def test_sqlite_db_file_and_dump_keep_backslashes_and_unicode(self):
        db = self.p("t.db")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE t (id INTEGER, text TEXT, label TEXT, note BLOB)")
        con.executemany("INSERT INTO t VALUES (?,?,?,?)", [
            (1, "C:\\temp\\new line\nsecond", "a", None),
            (2, "سلام, \"دنیا\" it's", "b", b"\x00\x01"),
            (3, None, "c", None)])
        con.commit()
        want = [["1", "C:\\temp\\new line\nsecond", "a", ""], ["2", "سلام, \"دنیا\" it's", "b", ""], ["3", "", "c", ""]]
        _, out = run(["--db", db, "--table", "t", "--out", self.p("a.csv")])
        self.assertEqual(read_csv(self.p("a.csv"))[1:], want)
        self.assertIn("BLOB", out)
        dump = self.write("t.sql", "\n".join(con.iterdump()))           # a real SQLite .dump
        self.assertEqual(S.detect_dialect(open(dump, encoding="utf-8").read()), "sqlite")
        run(["--sql", dump, "--table", "t", "--out", self.p("b.csv")])
        self.assertEqual(read_csv(self.p("b.csv"))[1:][:1] + read_csv(self.p("b.csv"))[3:], [want[0], want[2]])
        con.close()

    # ---- detection, encodings, errors
    def test_dialect_detection(self):
        self.assertEqual(S.detect_dialect(MYSQL_DUMP), "mysql")
        self.assertEqual(S.detect_dialect("CREATE TABLE t (a int);\nINSERT INTO t VALUES (1);"), "sqlite")
        self.assertEqual(S.detect_dialect("-- PostgreSQL database dump\nCOPY t (a) FROM stdin;\n1\n\\.\n"), "postgres")

    def test_postgres_dump_gives_clear_message(self):
        dump = self.write("pg.sql", "-- PostgreSQL database dump\nCOPY t (a) FROM stdin;\n1\n\\.\n")
        with self.assertRaises(SystemExit) as cm:
            run(["--sql", dump, "--list"])
        self.assertIn("PostgreSQL", str(cm.exception))
        self.assertIn("copy", str(cm.exception))

    def test_non_utf8_file_explains_encoding_option(self):
        sql = "CREATE TABLE t (a text);\nINSERT INTO t VALUES ('سلام دنيا');\n"   # Windows-1256 only has the Arabic yeh (ي), like real old dumps
        dump = self.write("old.sql", sql, enc="cp1256")
        with self.assertRaises(SystemExit) as cm:
            run(["--sql", dump, "--list"])
        self.assertIn("--encoding cp1256", str(cm.exception))
        run(["--sql", dump, "--encoding", "cp1256", "--table", "t", "--out", self.p("o.csv")])
        self.assertEqual(read_csv(self.p("o.csv")), [["a"], ["سلام دنيا"]])

    def test_utf8_bom_dump_is_accepted(self):
        dump = self.write("bom.sql", "\ufeffCREATE TABLE t (a text);\nINSERT INTO t VALUES ('x');\n")
        run(["--sql", dump, "--table", "t", "--out", self.p("o.csv")])
        self.assertEqual(read_csv(self.p("o.csv")), [["a"], ["x"]])

    def test_unknown_table_and_missing_out(self):
        dump = self.write("shop.sql", MYSQL_DUMP)
        with self.assertRaises(SystemExit) as cm:
            run(["--sql", dump, "--table", "nope", "--out", self.p("x.csv")])
        self.assertIn("comments", str(cm.exception))
        with self.assertRaises(SystemExit):
            run(["--sql", dump, "--table", "comments"])

    def test_unterminated_string_is_reported(self):
        dump = self.write("bad.sql", "CREATE TABLE t (a text);\nINSERT INTO t VALUES ('oops);\n")
        with self.assertRaises((ValueError, SystemExit)):
            run(["--sql", dump, "--list", "--dialect", "mysql"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
