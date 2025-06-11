#include <iomanip>
#include <iostream>
#include <sstream>
#include <arrow/array/array_base.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/table.h>

// #include "antlr4-runtime.h"
// #include "PipeQL/src/include/pipeql/parser/PipeQLParser.h"
// #include "PipeQL/src/include/pipeql/parser/PipeQLBaseVisitor.h"
// #include "lingodb/catalog/Catalog.h"
// #include "lingodb/compiler/frontend/PipeQL/Parser.h"

#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/Session.h"

using namespace lingodb;

// namespace interpreter::pipeql {

// class InterpreterPipeQLVisitor : public PipeQLBaseVisitor {
// public:
//     InterpreterPipeQLVisitor(lingodb::catalog::Catalog& catalog) : catalog(catalog) {}

//     antlrcpp::Any visitQuery(PipeQLParser::QueryContext* ctx) override {
//         auto fromResult = visit(ctx->fromClause());
//         try {
//             std::string tableName = std::any_cast<std::string>(fromResult);
//             currentTable = loadTable(tableName);
//         } catch (const std::bad_any_cast& e) {
//             return nullptr;
//         }

//         for (auto* op : ctx->pipeOperator()) {
//             auto result = visit(op);
//             try {
//                 currentTable = std::any_cast<std::shared_ptr<arrow::Table>>(result);
//             } catch (const std::bad_any_cast& e) {
//                 continue;
//             }
//         }

//         return currentTable;
//     }

//     antlrcpp::Any visitSelectOperator(PipeQLParser::SelectOperatorContext* ctx) override {
//         // Visit select expressions to get column names
//         std::vector<std::string> columns;
//         for (auto* expr : ctx->selectExpression()) {
//             auto result = visit(expr);
//             try {
//                 columns.push_back(std::any_cast<std::string>(result));
//             } catch (const std::bad_any_cast& e) {
//                 continue;
//             }
//         }

//         // Apply projection if columns specified
//         if (!columns.empty() && currentTable) {
//             std::vector<std::shared_ptr<arrow::Field>> fields;
//             std::vector<std::shared_ptr<arrow::Array>> arrays;
            
//             for (const auto& col : columns) {
//                 auto colIndex = currentTable->schema()->GetFieldIndex(col);
//                 if (colIndex == -1) continue;
                
//                 fields.push_back(currentTable->schema()->field(colIndex));
//                 arrays.push_back(currentTable->column(colIndex));
//             }
            
//             auto schema = arrow::schema(fields);
//             currentTable = arrow::Table::Make(schema, arrays);
//         }

//         return currentTable;
//     }

//     antlrcpp::Any visitWhereOperator(PipeQLParser::WhereOperatorContext* ctx) override {
//         auto boolExpr = visit(ctx->booleanExpression());
//         try {
//             auto [col, value] = std::any_cast<std::pair<std::string, std::string>>(boolExpr);
            
//             // Apply filter
//             if (currentTable) {
//                 auto filteredTable = applyFilter(currentTable, col, value);
//                 currentTable = filteredTable;
//             }
//         } catch (const std::bad_any_cast& e) {
//             return nullptr;
//         }
//         return currentTable;
//     }

//     antlrcpp::Any visitFromClause(PipeQLParser::FromClauseContext* ctx) override {
//         if (ctx->IDENTIFIER()) {
//             return ctx->IDENTIFIER()->getText();
//         }
//         return nullptr;
//     }

//     antlrcpp::Any visitPipeOperator(PipeQLParser::PipeOperatorContext* ctx) override {
//         if (ctx->selectOperator()) {
//             return visit(ctx->selectOperator());
//         }
//         if (ctx->whereOperator()) {
//             return visit(ctx->whereOperator());
//         }
//         if (ctx->orderByOperator()) {
//             return visit(ctx->orderByOperator());
//         }
//         if (ctx->unionOperator()) {
//             return visit(ctx->unionOperator());
//         }
//         if (ctx->intersectOperator()) {
//             return visit(ctx->intersectOperator());
//         }
//         if (ctx->exceptOperator()) {
//             return visit(ctx->exceptOperator());
//         }
//         if (ctx->assertOperator()) {
//             return visit(ctx->assertOperator());
//         }
//         if (ctx->limitClause()) {
//             return visit(ctx->limitClause());
//         }
//         return nullptr;
//     }

//     antlrcpp::Any visitOrderByOperator(PipeQLParser::OrderByOperatorContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitUnionOperator(PipeQLParser::UnionOperatorContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitIntersectOperator(PipeQLParser::IntersectOperatorContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitExceptOperator(PipeQLParser::ExceptOperatorContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitAssertOperator(PipeQLParser::AssertOperatorContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitLimitClause(PipeQLParser::LimitClauseContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitOffsetClause(PipeQLParser::OffsetClauseContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitSelectExpression(PipeQLParser::SelectExpressionContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitOrderExpression(PipeQLParser::OrderExpressionContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitBooleanExpression(PipeQLParser::BooleanExpressionContext* ctx) override {
//         if (ctx->comparisonOperator()) {
//             auto op = visit(ctx->comparisonOperator());
//             auto left = visit(ctx->expression(0));
//             auto right = visit(ctx->expression(1));

//             try {
//                 std::string opStr = std::any_cast<std::string>(op);
//                 std::string leftStr = std::any_cast<std::string>(left);
//                 std::string rightStr = std::any_cast<std::string>(right);
//                 return std::make_pair(leftStr, rightStr);
//             } catch (const std::bad_any_cast& e) {
//                 return nullptr;
//             }
//         }
//         return nullptr;
//     }

//     antlrcpp::Any visitComparisonOperator(PipeQLParser::ComparisonOperatorContext* ctx) override {
//         if (ctx->getText() == "=") return "==";
//         if (ctx->getText() == "!=") return "!=";
//         if (ctx->getText() == "<") return "<";
//         if (ctx->getText() == "<=") return "<=";
//         if (ctx->getText() == ">") return ">";
//         if (ctx->getText() == ">=") return ">=";
//         return "==";
//     }

//     antlrcpp::Any visitPayloadExpression(PipeQLParser::PayloadExpressionContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitExpression(PipeQLParser::ExpressionContext* ctx) override {
//         if (ctx->IDENTIFIER()) {
//             return ctx->IDENTIFIER()->getText();
//         }
//         if (ctx->literal()) {
//             return visit(ctx->literal());
//         }
//         return nullptr;
//     }

//     antlrcpp::Any visitFunctionCall(PipeQLParser::FunctionCallContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

//     antlrcpp::Any visitLiteral(PipeQLParser::LiteralContext* ctx) override {
//         if (ctx->STRING()) {
//             std::string text = ctx->STRING()->getText();
//             // Remove quotes
//             return text.substr(1, text.length() - 2);
//         }
//         if (ctx->NUMBER()) {
//             return ctx->NUMBER()->getText();
//         }
//         return nullptr;
//     }

//     antlrcpp::Any visitAliasClause(PipeQLParser::AliasClauseContext* ctx) override {
//         // Implementation needed
//         return nullptr;
//     }

// private:
//     lingodb::catalog::Catalog& catalog;
//     std::shared_ptr<arrow::Table> currentTable;

//     std::shared_ptr<arrow::Table> loadTable(const std::string& tableName) {
//         auto table = catalog.getTypedEntry<catalog::TableCatalogEntry>(tableName);
//         if (!table) return nullptr;
//         return table;
//     }

//     std::shared_ptr<arrow::Table> applyFilter(const std::shared_ptr<arrow::Table>& table,
//                                             const std::string& column,
//                                             const std::string& value) {
//         if (!table) return nullptr;

//         // Get column index
//         auto colIndex = table->schema()->GetFieldIndex(column);
//         if (colIndex == -1) return nullptr;

//         // Create filter array
//         std::vector<bool> filter;
//         auto array = table->column(colIndex);
        
//         for (int64_t i = 0; i < array->length(); i++) {
//             if (array->IsNull(i)) {
//                 filter.push_back(false);
//                 continue;
//             }

//             std::string rowValue;
//             if (array->type_id() == arrow::Type::STRING) {
//                 auto stringArray = std::static_pointer_cast<arrow::StringArray>(array);
//                 rowValue = stringArray->GetString(i);
//             } else {
//                 // Convert numeric values to string for comparison
//                 rowValue = array->GetScalar(i).ValueOrDie()->ToString();
//             }

//             filter.push_back(compareValues(rowValue, value, "=="));
//         }

//         // Apply filter
//         auto filterArray = arrow::BooleanArray::FromValues(filter).ValueOrDie();
//         auto options = arrow::compute::FilterOptions::Defaults();
//         auto result = arrow::compute::Filter(table, filterArray, options);
//         return result.ValueOrDie().table();
//     }

//     bool compareValues(const std::string& value1, const std::string& value2, const std::string& op) {
//         // Try numeric comparison first
//         try {
//             double num1 = std::stod(value1);
//             double num2 = std::stod(value2);
            
//             if (op == "==") return num1 == num2;
//             if (op == "!=") return num1 != num2;
//             if (op == "<") return num1 < num2;
//             if (op == "<=") return num1 <= num2;
//             if (op == ">") return num1 > num2;
//             if (op == ">=") return num1 >= num2;
//         } catch (...) {
//             // Fall back to string comparison if numeric conversion fails
//             if (op == "==") return value1 == value2;
//             if (op == "!=") return value1 != value2;
//             if (op == "<") return value1 < value2;
//             if (op == "<=") return value1 <= value2;
//             if (op == ">") return value1 > value2;
//             if (op == ">=") return value1 >= value2;
//         }
//         return false;
//     }
// };
// } // namespace interpreter::pipeql

namespace {
unsigned char hexval(unsigned char c) {
    if ('0' <= c && c <= '9')
        return c - '0';
    else if ('a' <= c && c <= 'f')
        return c - 'a' + 10;
    else if ('A' <= c && c <= 'F')
        return c - 'A' + 10;
    else
        abort();
}

void printTable(const std::shared_ptr<arrow::Table>& table) {
    if (table->columns().empty()) {
        std::cout << "Statement executed successfully." << std::endl;
        return;
    }

    std::vector<std::string> columnReps;
    std::vector<size_t> positions;
    arrow::PrettyPrintOptions options;
    options.indent_size = 0;
    options.window = 100;
    std::cout << "|";
    std::string rowSep = "-";
    std::vector<bool> convertHex;
    for (auto c : table->columns()) {
        std::cout << std::setw(30) << table->schema()->field(positions.size())->name() << "  |";
        convertHex.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY);
        rowSep += std::string(33, '-');
        std::stringstream sstr;
        auto status = arrow::PrettyPrint(*c.get(), options, &sstr);
        if (!status.ok()) {
            std::cerr << "Error printing column: " << status.ToString() << std::endl;
            return;
        }
        columnReps.push_back(sstr.str());
        positions.push_back(0);
    }
    std::cout << std::endl
              << rowSep << std::endl;
    bool cont = true;
    while (cont) {
        cont = false;
        bool skipNL = false;
        for (size_t column = 0; column < columnReps.size(); column++) {
            char lastHex = 0;
            bool first = true;
            std::stringstream out;
            while (positions[column] < columnReps[column].size()) {
                cont = true;
                char curr = columnReps[column][positions[column]];
                char next = columnReps[column][positions[column] + 1];
                positions[column]++;
                if (first && (curr == '[' || curr == ']' || curr == ',')) {
                    continue;
                }
                if (curr == ',' && next == '\n') {
                    continue;
                }
                if (curr == '\n') {
                    break;
                } else {
                    if (convertHex[column]) {
                        if (isxdigit(curr)) {
                            if (lastHex == 0) {
                                first = false;
                                lastHex = curr;
                            } else {
                                char converted = (hexval(lastHex) << 4 | hexval(curr));
                                out << converted;
                                lastHex = 0;
                            }
                        } else {
                            first = false;
                            out << curr;
                        }
                    } else {
                        first = false;
                        out << curr;
                    }
                }
            }
            if (first) {
                skipNL = true;
            } else {
                if (column == 0) {
                    std::cout << "|";
                }
                std::cout << std::setw(30) << out.str() << "  |";
            }
        }
        if (!skipNL) {
            std::cout << "\n";
        }
    }
}

std::string getValueFromArray(const std::shared_ptr<arrow::Array>& array, int64_t index) {
    if (array->IsNull(index)) {
        return "NULL";
    }

    switch (array->type_id()) {
        case arrow::Type::INT8:
            return std::to_string(std::static_pointer_cast<arrow::Int8Array>(array)->Value(index));
        case arrow::Type::INT16:
            return std::to_string(std::static_pointer_cast<arrow::Int16Array>(array)->Value(index));
        case arrow::Type::INT32:
            return std::to_string(std::static_pointer_cast<arrow::Int32Array>(array)->Value(index));
        case arrow::Type::INT64:
            return std::to_string(std::static_pointer_cast<arrow::Int64Array>(array)->Value(index));
        case arrow::Type::FLOAT:
            return std::to_string(std::static_pointer_cast<arrow::FloatArray>(array)->Value(index));
        case arrow::Type::DOUBLE:
            return std::to_string(std::static_pointer_cast<arrow::DoubleArray>(array)->Value(index));
        case arrow::Type::STRING:
        case arrow::Type::BINARY:
            return std::static_pointer_cast<arrow::StringArray>(array)->GetString(index);
        case arrow::Type::BOOL:
            return std::static_pointer_cast<arrow::BooleanArray>(array)->Value(index) ? "true" : "false";
        default:
            return "UNSUPPORTED TYPE";
    }
}

// Limitation: Try to convert to numbers for numeric comparison
bool compareValues(const std::string& value1, const std::string& value2, const std::string& op) {
    try {
        double num1 = std::stod(value1);
        double num2 = std::stod(value2);

        if (op == "==") return num1 == num2;
        if (op == "!=") return num1 != num2;
        if (op == ">") return num1 > num2;
        if (op == "<") return num1 < num2;
        if (op == ">=") return num1 >= num2;
        if (op == "<=") return num1 <= num2;
    } catch (...) {
        // If conversion fails, fall back to string comparison
        if (op == "==") return value1 == value2;
        if (op == "!=") return value1 != value2;
        if (op == ">") return value1 > value2;
        if (op == "<") return value1 < value2;
        if (op == ">=") return value1 >= value2;
        if (op == "<=") return value1 <= value2;
    }
    return false;
}

std::shared_ptr<arrow::RecordBatch> processBatch(const std::shared_ptr<arrow::RecordBatch>& batch,
                                                 const std::vector<std::string>& selectedColumns,
                                                 const std::string& filterColumn,
                                                 const std::string& filterValue,
                                                 const std::string& filterOp = "==") {
    std::vector<int> colIndices;
    if (selectedColumns.empty()) {
        for (int i = 0; i < batch->num_columns(); i++) {
            colIndices.push_back(i);
        }
    } else {
        for (const auto& colName : selectedColumns) {
            int idx = batch->schema()->GetFieldIndex(colName);
            if (idx >= 0) {
                colIndices.push_back(idx);
            }
        }
    }

    int filterColIdx = -1;
    if (!filterColumn.empty()) {
        filterColIdx = batch->schema()->GetFieldIndex(filterColumn);
    }

    std::vector<std::shared_ptr<arrow::Array>> filteredArrays;
    filteredArrays.reserve(colIndices.size());
    std::vector<std::shared_ptr<arrow::Field>> filteredFields;
    filteredFields.reserve(colIndices.size());

    std::vector<int64_t> matchingRows;
    matchingRows.reserve(batch->num_rows());
    for (int64_t row = 0; row < batch->num_rows(); row++) {
        if (filterColIdx >= 0) {
            auto filterArray = batch->column(filterColIdx);
            std::string rowValue = getValueFromArray(filterArray, row);
            if (!compareValues(rowValue, filterValue, filterOp)) {
                continue; // Skip this row if it doesn't match the filter
            }
        }
        matchingRows.push_back(row);
    }

    arrow::Int64Builder builder;
    if (!builder.AppendValues(matchingRows).ok()) {
        std::cerr << "Error creating indices array" << std::endl;
        return nullptr;
    }
    std::shared_ptr<arrow::Int64Array> indicesArray;
    if (!builder.Finish(&indicesArray).ok()) {
        std::cerr << "Error finalizing indices array" << std::endl;
        return nullptr;
    }

    // Second pass: create filtered arrays for each selected column
    for (int colIdx : colIndices) {
        auto array = batch->column(colIdx);
        auto field = batch->schema()->field(colIdx);

        // Create a new array with only the matching rows
        std::shared_ptr<arrow::Array> filteredArray;
        switch (array->type_id()) {
            case arrow::Type::INT8: {
                arrow::Int8Builder builder;
                for (int64_t row : matchingRows) {
                    if (array->IsNull(row)) {
                        if (!builder.AppendNull().ok()) {
                            std::cerr << "Error appending null value" << std::endl;
                            return nullptr;
                        }
                    } else {
                        if (!builder.Append(std::static_pointer_cast<arrow::Int8Array>(array)->Value(row)).ok()) {
                            std::cerr << "Error appending value" << std::endl;
                            return nullptr;
                        }
                    }
                }
                if (!builder.Finish(&filteredArray).ok()) {
                    std::cerr << "Error finishing builder" << std::endl;
                    return nullptr;
                }
                break;
            }
            case arrow::Type::INT16: {
                arrow::Int16Builder builder;
                for (int64_t row : matchingRows) {
                    if (array->IsNull(row)) {
                        if (!builder.AppendNull().ok()) {
                            std::cerr << "Error appending null value" << std::endl;
                            return nullptr;
                        }
                    } else {
                        if (!builder.Append(std::static_pointer_cast<arrow::Int16Array>(array)->Value(row)).ok()) {
                            std::cerr << "Error appending value" << std::endl;
                            return nullptr;
                        }
                    }
                }
                if (!builder.Finish(&filteredArray).ok()) {
                    std::cerr << "Error finishing builder" << std::endl;
                    return nullptr;
                }
                break;
            }
            case arrow::Type::INT32: {
                arrow::Int32Builder builder;
                for (int64_t row : matchingRows) {
                    if (array->IsNull(row)) {
                        if (!builder.AppendNull().ok()) {
                            std::cerr << "Error appending null value" << std::endl;
                            return nullptr;
                        }
                    } else {
                        if (!builder.Append(std::static_pointer_cast<arrow::Int32Array>(array)->Value(row)).ok()) {
                            std::cerr << "Error appending value" << std::endl;
                            return nullptr;
                        }
                    }
                }
                if (!builder.Finish(&filteredArray).ok()) {
                    std::cerr << "Error finishing builder" << std::endl;
                    return nullptr;
                }
                break;
            }
            case arrow::Type::INT64: {
                arrow::Int64Builder builder;
                for (int64_t row : matchingRows) {
                    if (array->IsNull(row)) {
                        if (!builder.AppendNull().ok()) {
                            std::cerr << "Error appending null value" << std::endl;
                            return nullptr;
                        }
                    } else {
                        if (!builder.Append(std::static_pointer_cast<arrow::Int64Array>(array)->Value(row)).ok()) {
                            std::cerr << "Error appending value" << std::endl;
                            return nullptr;
                        }
                    }
                }
                if (!builder.Finish(&filteredArray).ok()) {
                    std::cerr << "Error finishing builder" << std::endl;
                    return nullptr;
                }
                break;
            }
            case arrow::Type::STRING: {
                arrow::StringBuilder builder;
                for (int64_t row : matchingRows) {
                    if (array->IsNull(row)) {
                        if (!builder.AppendNull().ok()) {
                            std::cerr << "Error appending null value" << std::endl;
                            return nullptr;
                        }
                    } else {
                        if (!builder.Append(std::static_pointer_cast<arrow::StringArray>(array)->GetString(row)).ok()) {
                            std::cerr << "Error appending value" << std::endl;
                            return nullptr;
                        }
                    }
                }
                if (!builder.Finish(&filteredArray).ok()) {
                    std::cerr << "Error finishing builder" << std::endl;
                    return nullptr;
                }
                break;
            }
            default: {
                std::cerr << "Unsupported type for filtering: " << array->type()->ToString() << std::endl;
                return nullptr;
            }
        }

        filteredArrays.push_back(filteredArray);
        filteredFields.push_back(field);
    }

    auto schema = arrow::schema(filteredFields);
    return arrow::RecordBatch::Make(schema, matchingRows.size(), filteredArrays);
}
} // anonymous namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::cerr << "Usage: " << argv[0] << " <db_dir> <table_name> [columns] [filter_column] [filter_value] [filter_op]" << std::endl;
        std::cerr << "  columns: comma-separated list of column names (optional)" << std::endl;
        std::cerr << "  filter_column: column name to filter on (optional)" << std::endl;
        std::cerr << "  filter_value: value to filter for (optional)" << std::endl;
        std::cerr << "  filter_op: comparison operator (==, !=, >, <, >=, <=) (optional, defaults to ==)" << std::endl;
        return 1;
    }

    std::string dbDir = argv[1];
    std::string tableName = argv[2];

    std::vector<std::string> selectedColumns;
    std::string filterColumn;
    std::string filterValue;
    std::string filterOp = "=="; // Default operator

    if (argc > 3) {
        std::string columns = argv[3];
        std::stringstream ss(columns);
        std::string column;
        while (std::getline(ss, column, ',')) {
            column.erase(0, column.find_first_not_of(" \t\r\n"));
            column.erase(column.find_last_not_of(" \t\r\n") + 1);
            if (!column.empty()) {
                selectedColumns.push_back(column);
            }
        }
    }

    if (argc > 4) {
        filterColumn = argv[4];
        filterColumn.erase(0, filterColumn.find_first_not_of(" \t\r\n"));
        filterColumn.erase(filterColumn.find_last_not_of(" \t\r\n") + 1);
    }

    if (argc > 5) {
        filterValue = argv[5];
        filterValue.erase(0, filterValue.find_first_not_of(" \t\r\n"));
        filterValue.erase(filterValue.find_last_not_of(" \t\r\n") + 1);
    }

    if (argc > 6) {
        filterOp = argv[6];
        filterOp.erase(0, filterOp.find_first_not_of(" \t\r\n"));
        filterOp.erase(filterOp.find_last_not_of(" \t\r\n") + 1);

        // Validate operator
        if (filterOp != "==" && filterOp != "!=" && filterOp != ">" &&
            filterOp != "<" && filterOp != ">=" && filterOp != "<=") {
            std::cerr << "Error: Invalid operator. Must be one of: ==, !=, >, <, >=, <=" << std::endl;
            return 1;
        }
    }

    if (!filterColumn.empty() && filterValue.empty()) {
        std::cerr << "Error: filter_value must be provided when filter_column is specified" << std::endl;
        return 1;
    }

    try {
        auto session = runtime::Session::createSession(dbDir, true);
        auto catalog = session->getCatalog();

        auto tableEntry = catalog->getTypedEntry<catalog::TableCatalogEntry>(tableName);
        if (!tableEntry) {
            std::cerr << "Table '" << tableName << "' not found in database." << std::endl;
            return 1;
        }

        std::string arrowFile = dbDir + "/" + tableName + ".arrow";

        auto infile = arrow::io::ReadableFile::Open(arrowFile);
        if (!infile.ok()) {
            std::cerr << "Failed to open Arrow file: " << infile.status().ToString() << std::endl;
            return 1;
        }

        auto reader = arrow::ipc::RecordBatchFileReader::Open(*infile);
        if (!reader.ok()) {
            std::cerr << "Failed to open Arrow reader: " << reader.status().ToString() << std::endl;
            return 1;
        }

        std::vector<std::shared_ptr<arrow::RecordBatch>> filteredBatches;
        for (int i = 0; i < (*reader)->num_record_batches(); i++) {
            auto batch = (*reader)->ReadRecordBatch(i);
            if (!batch.ok()) {
                std::cerr << "Failed to read batch " << i << ": " << batch.status().ToString() << std::endl;
                return 1;
            }
            auto filteredBatch = processBatch(*batch, selectedColumns, filterColumn, filterValue, filterOp);
            if (filteredBatch) {
                filteredBatches.push_back(filteredBatch);
            }
        }

        auto table = arrow::Table::FromRecordBatches(filteredBatches);
        if (!table.ok()) {
            std::cerr << "Failed to create table from batches: " << table.status().ToString() << std::endl;
            return 1;
        }

        printTable(*table);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}