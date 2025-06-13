<div align="center">
  <img src="https://www.lingo-db.com/images/lingodb-black-title.png" height="50">
</div>
<p>&nbsp;</p>
<p align="center">
  <a href="https://github.com/lingo-db/lingo-db/actions">
    <img src="https://github.com/lingo-db/lingo-db/actions/workflows/workflow.yml/badge.svg?branch=main" alt="Github Actions Badge">
  </a>
  <a href="https://codecov.io/gh/lingo-db/lingo-db" >
    <img src="https://codecov.io/gh/lingo-db/lingo-db/branch/main/graph/badge.svg?token=7RC3UD5YEA"/>
  </a>
</p>

# LingoDB
LingoDB is a cutting-edge data processing system that leverages compiler technology to achieve unprecedented flexibility and extensibility without sacrificing performance. It supports a wide range of data-processing workflows beyond relational SQL queries, thanks to declarative sub-operators. Furthermore, LingoDB can perform cross-domain optimization by interleaving optimization passes of different domains and its flexibility enables sustainable support for heterogeneous hardware.

# Using LingoDB
You can try out LingoDB through different ways:
1. Use the hosted [SQL Webinterface](https://www.lingo-db.com/interface/)
2. Use the python package: `pip install lingodb`
3. Build it yourself by following the [documentation](https://www.lingo-db.com/docs/gettingstarted/install/#building-from-source)

## Documentation
For LingoDB's documentation, please visit [the documentation website](https://www.lingo-db.com/docs/) based on [this github repo](https://github.com/lingo-db/lingo-db.github.io).

## Contributing
Before contributing, please first read the [contribution guidelines](https://www.lingo-db.com/docs/next/ForDevelopers/Contributing).


# PipeQL Project Implementation

## Instructions
### SQL Execution (MLIR)
 - Executing the SQL Engine `SELECT *` Operation:
 ```bash
 ./build/lingodb-debug/run-sql ./docs/test_sql_1.sql test_dir
 ```
 - Executing the SQL Engine `SELECT` with Projection Operation:
 ```bash
 ./build/lingodb-debug/run-sql ./docs/test_sql_2.sql test_dir
 ```
 - Executing the SQL Engine `SELECT` with Where Operation:
 ```bash
  ./build/lingodb-debug/run-sql ./docs/test_sql_3.sql test_dir
 ```

### PipeQL Execution (MLIR)
 - Executing the PipeQL Engine `SELECT *` Operation:
 ```bash
 ./build/lingodb-debug/pipeql-to-mlir ./docs/test_pipeql_1.sql test_dir > ./docs/pipeql1.mlir
 ./build/lingodb-debug/run-mlir ./docs/pipeql1.mlir test_dir
 ```
 - Executing the PipeQL Engine `SELECT` with Projection Operation:
 ```bash
 ./build/lingodb-debug/pipeql-to-mlir ./docs/test_pipeql_2.sql test_dir > ./docs/pipeql2.mlir
 ./build/lingodb-debug/run-mlir ./docs/pipeql2.mlir test_dir
 ```
 - Executing the PipeQL Engine `SELECT` with Where Operation:
 ```bash
 ./build/lingodb-debug/pipeql-to-mlir ./docs/test_pipeql_3.sql test_dir > ./docs/pipeql3.mlir
 ./build/lingodb-debug/run-mlir ./docs/pipeql3.mlir test_dir
 ```

### PipeQL Execution (Interpreted)
 - Executing the PipeQL Engine `SELECT *` Operation:
 ```bash
 ./build/lingodb-debug/query-table test_dir "FROM users |> SELECT *"
 ```
 - Executing the PipeQL Engine `SELECT` with Projection Operation:
 ```bash
  ./build/lingodb-debug/query-table test_dir "FROM users |> SELECT name,age,email"
 ```
 - Executing the PipeQL Engine `SELECT` with Where Operation:
 ```bash
 ./build/lingodb-debug/query-table test_dir "FROM users |> SELECT * |> WHERE age > 18"
 ```

## Results

### Execution times
TODO: Update
<!-- | Operation | Description | SQL Example | PipeQL Example | SQL Engine (MLIR) (ms)  | PipeQL Engine (MLIR) (ms) | PipeQL Engine (Interpreted) (ms)
|-----------|-------------|---------| ------- | ---------| ----------| --------------------|
| SELECT * | Retrieves all columns from the specified table | `SELECT * FROM users` | `FROM users \|> SELECT *` | 50.15 | 47.15 | 522.2
| SELECT with Projection | Retrieves specific columns from the table | `SELECT name, email FROM users` | `FROM users \|> SELECT name,email` | 44.32 | 47.95  | 359.2
| SELECT with WHERE | Filters rows based on specified conditions | `SELECT * FROM users WHERE age > 18` | `FROM users \|> SELECT name,email \|> WHERE age > 18` | 59.91 | 61.51 | 3283.8 -->


### Comilation times

| Operation | Description | SQL Example | PipeQL Example | SQL Parser  | PipeQL Parser
|-----------|-------------|---------| ------- | ---------| ----------| 
| SELECT * | Retrieves all columns from the specified table | `SELECT * FROM users` | `FROM users \|> SELECT *` | 3,478.8 | 4,311
| SELECT with Projection | Retrieves specific columns from the table | `SELECT name, email FROM users` | `FROM users \|> SELECT name,email` | 3,472.4 | 4,564.8  
| SELECT with WHERE | Filters rows based on specified conditions | `SELECT * FROM users WHERE age > 18` | `FROM users \|> SELECT name,email \|> WHERE age > 18` | 3,764.8 | 5,824.6 