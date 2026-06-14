"""Tests for evaluate.py — SQL extraction, execution, timeout."""
from projects.nlp.text2sql_finetune.evaluate import (
    extract_sql,
    parse_create_table,
    execute_sql,
    execute_with_timeout,
)

ASSISTANT_OUTPUT = """<|im_start|>system
You are a SQL expert.
<|im_end|>
<|im_start|>user
Schema:
CREATE TABLE stadium (id int, name text, capacity int);

Question: total capacity?
<|im_end|>
<|im_start|>assistant
SELECT SUM(capacity) FROM stadium
<|im_end|>"""


def test_extract_sql_from_assistant_tags():
    sql = extract_sql(ASSISTANT_OUTPUT)
    assert sql == "SELECT SUM(capacity) FROM stadium", f"Got: {sql}"


def test_extract_sql_multiple_tags():
    text = "<|im_start|>assistant\nSELECT 1<|im_end|>\n<|im_start|>assistant\nSELECT 2<|im_end|>"
    sql = extract_sql(text)
    assert sql == "SELECT 2", f"Got: {sql}"  # Last assistant block


def test_parse_create_table_single():
    context = "CREATE TABLE t1 (a int, b text);"
    tables = parse_create_table(context)
    assert len(tables) == 1
    assert "CREATE TABLE t1" in tables[0]


def test_parse_create_table_multiple():
    context = "CREATE TABLE t1 (a int);\nSome text\nCREATE TABLE t2 (x real);"
    tables = parse_create_table(context)
    assert len(tables) == 2


def test_parse_create_table_case_insensitive():
    context = "create table t1 (a int);"
    tables = parse_create_table(context)
    assert len(tables) == 1


def test_execute_sql_correct():
    create_tables = ["CREATE TABLE t (x int, y text)", "INSERT INTO t VALUES (1, 'a'), (2, 'b')"]
    success, result = execute_sql(create_tables, "SELECT x FROM t WHERE y = 'a'")
    assert success
    assert result == [(1,)], f"Got: {result}"


def test_execute_sql_syntax_error():
    success, result = execute_sql(["CREATE TABLE t (x int)"], "SELEC x FROM t")
    assert not success


def test_execute_sql_table_not_found():
    success, result = execute_sql(["CREATE TABLE t (x int)"], "SELECT x FROM nonexistent")
    assert not success


def test_execute_sql_timeout():
    """A slow Cartesian product should time out."""
    create_tables = ["CREATE TABLE a (x int)", "CREATE TABLE b (y int)"]
    for i in range(100):
        create_tables.append(f"INSERT INTO a VALUES ({i})")
        create_tables.append(f"INSERT INTO b VALUES ({i})")

    success, result = execute_with_timeout(
        create_tables,
        "SELECT COUNT(*) FROM a CROSS JOIN b CROSS JOIN a c CROSS JOIN b d",
        timeout=0.5,
    )
    assert not success
    assert result == "__TIMEOUT__", f"Got: {result}"
