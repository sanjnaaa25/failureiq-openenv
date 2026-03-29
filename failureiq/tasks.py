from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    log: str
    root_cause_keywords: List[str]
    category: str
    misleading_keywords: List[str]
    code_snippet: Optional[str]
    config_snippet: Optional[str]
    data_context: Optional[str]
    fix_keywords: List[str]
    validation_steps: List[str]


TASKS: List[TaskSpec] = [
    TaskSpec(
        task_id="code_null_pointer",
        difficulty="easy",
        log=(
            "INFO Starting job...\n"
            "WARN Retrying execution...\n"
            "INFO Pipeline stage: enrich-user-profiles\n"
            "INFO Writing 128 records to user_profile table\n"
            "\n"
            "java.lang.RuntimeException: Job failed\n"
            "    at Executor.run(Executor.java:88)\n"
            "\n"
            "Caused by: java.util.concurrent.ExecutionException\n"
            "    at java.util.concurrent.FutureTask.get(FutureTask.java:205)\n"
            "\n"
            "Caused by: java.lang.NullPointerException\n"
            "    at DataProcessor.processBatch(DataProcessor.java:45)\n"
        ),
        root_cause_keywords=["nullpointer", "null pointer", "null"],
        category="code",
        misleading_keywords=["retry", "job failed"],
        code_snippet=(
            "File: DataProcessor.java\n"
            "\n"
            "class DataProcessor {\n"
            "    public void processBatch(List<Record> records) {\n"
            "        for (Record record : records) {\n"
            "            String region = record.getMetadata().get(\"region\");\n"
            "            if (region.equals(\"APAC\")) {\n"
            "                enrichRegion(record);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
        ),
        config_snippet=None,
        data_context=None,
        fix_keywords=["null check", "null-safe", "if (region != null", "optional", "guard"],
        validation_steps=[
            "Add a null guard before region.equals(...)",
            "Add unit test where metadata is missing",
            "Re-run batch with sample records",
        ],
    ),
    TaskSpec(
        task_id="config_oom",
        difficulty="medium",
        log=(
            "INFO Running Spark job...\n"
            "ERROR Job failed due to timeout\n"
            "WARN Task retry...\n"
            "INFO Stage 4/8: join-events-with-users\n"
            "\n"
            "java.lang.RuntimeException: Executor lost\n"
            "    at org.apache.spark.ExecutorRunner.run(ExecutorRunner.scala:112)\n"
            "\n"
            "Caused by: java.lang.OutOfMemoryError: Java heap space\n"
            "    at org.apache.spark.util.collection.AppendOnlyMap.insert(AppendOnlyMap.scala:155)\n"
        ),
        root_cause_keywords=["outofmemory", "out of memory", "oom", "heap"],
        category="config",
        misleading_keywords=["timeout", "executor lost", "retry"],
        code_snippet=None,
        config_snippet=(
            "File: spark-defaults.conf\n"
            "\n"
            "spark.executor.memory=2g\n"
            "spark.executor.cores=1\n"
            "spark.sql.shuffle.partitions=200\n"
        ),
        data_context=None,
        fix_keywords=["increase", "memory", "executor", "4g", "6g", "8g"],
        validation_steps=[
            "Increase spark.executor.memory",
            "Re-run job on the same input sample",
            "Monitor executor memory usage",
        ],
    ),
    TaskSpec(
        task_id="data_constraint",
        difficulty="hard",
        log=(
            "INFO Starting batch insert...\n"
            "ERROR Batch insert failed\n"
            "\n"
            "org.springframework.dao.DataIntegrityViolationException: could not execute statement;\n"
            "    nested exception is org.postgresql.util.PSQLException\n"
            "\n"
            "Caused by: org.postgresql.util.PSQLException: ERROR: value too long for type character varying(30)\n"
            "    at org.postgresql.core.v3.QueryExecutorImpl.receiveErrorResponse(QueryExecutorImpl.java:2553)\n"
        ),
        root_cause_keywords=["value too long", "varchar", "constraint", "data too long", "length"],
        category="data",
        misleading_keywords=["batch insert failed", "statement"],
        code_snippet=None,
        config_snippet=None,
        data_context=(
            "Table: customer_profile\n"
            "Column: last_name VARCHAR(30)\n"
            "Sample payload: {\"last_name\": \"Montgomery-Johnson-Smythe\"}\n"
        ),
        fix_keywords=["truncate", "validate", "schema", "varchar(30)", "length"],
        validation_steps=[
            "Add input validation for length <= 30",
            "Backfill with truncated values or widen column",
            "Re-run insert for failed records",
        ],
    ),
]
