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

#include "PipeQL/src/include/pipeql/parser/PipeQLBaseVisitor.h"
#include "PipeQL/src/include/pipeql/parser/PipeQLLexer.h"
#include "PipeQL/src/include/pipeql/parser/PipeQLParser.h"
#include "lingodb/catalog/Catalog.h"

#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/runtime/Session.h"

using namespace lingodb;

namespace interpreter::pipeql {

class InterpreterPipeQLVisitor : public PipeQLBaseVisitor {
   public:
   InterpreterPipeQLVisitor(lingodb::catalog::Catalog& catalog, std::string dbDir) : catalog(catalog), dbDir(dbDir) {}

   antlrcpp::Any visitQuery(PipeQLParser::QueryContext* ctx) override {
      auto fromResult = visit(ctx->fromClause());
      try {
         std::string tableName = std::any_cast<std::string>(fromResult);
         currentTable = loadTable(tableName);
         if (!currentTable) {
            std::cerr << "Table '" << tableName << "' not found in database." << std::endl;
            return nullptr;
         }
      } catch (const std::bad_any_cast& e) {
         return nullptr;
      }

      for (auto* op : ctx->pipeOperator()) {
         if (op->whereOperator()) {
            auto result = visitWhereOperator(op->whereOperator());
            try {
               currentTable = std::any_cast<std::shared_ptr<arrow::Table>>(result);
            } catch (const std::bad_any_cast& e) {
               continue;
            }
         }
         if (op->selectOperator()) {
            auto result = visitSelectOperator(op->selectOperator());
            try {
               currentTable = std::any_cast<std::shared_ptr<arrow::Table>>(result);
            } catch (const std::bad_any_cast& e) {
               continue;
            }
         }
      }

      return currentTable;
   }

   antlrcpp::Any visitSelectOperator(PipeQLParser::SelectOperatorContext* ctx) override {
      if (!currentTable) return nullptr;

      std::vector<std::string> columns;
      for (auto* expr : ctx->selectExpression()) {
         std::string column;
         if (expr->expression()) {
            column = expr->expression()->getText();
         } else {
            column = expr->getText();
         }
         columns.push_back(column);
      }

      if (!columns.empty()) {
         std::vector<std::shared_ptr<arrow::Field>> fields;
         std::vector<std::shared_ptr<arrow::Array>> arrays;

         for (const auto& col : columns) {
            auto colIndex = currentTable->schema()->GetFieldIndex(col);
            if (colIndex == -1) continue;

            fields.push_back(currentTable->schema()->field(colIndex));
            auto chunkedArray = currentTable->column(colIndex);
            arrays.push_back(chunkedArray->chunk(0));
         }

         auto schema = arrow::schema(fields);
         currentTable = arrow::Table::Make(schema, arrays);
      }

      return currentTable;
   }

   antlrcpp::Any visitWhereOperator(PipeQLParser::WhereOperatorContext* ctx) override {
      if (!currentTable) return nullptr;

      auto* boolExpr = ctx->booleanExpression();
      std::string col = boolExpr->expression(0)->getText();
      std::string value = boolExpr->expression(1)->getText();

      std::string op = boolExpr->comparisonOperator()->getText();
      if (op.find("==") != std::string::npos)
         op = "=="; // Convert single = to == for comparison
      else if (op.find("!=") != std::string::npos)
         op = "!="; // Convert single = to == for comparison
      else if (op.find("<") != std::string::npos)
         op = "<"; // Convert single = to == for comparison
      else if (op.find("<=") != std::string::npos)
         op = "<="; // Convert single = to == for comparison
      else if (op.find(">") != std::string::npos)
         op = ">"; // Convert single = to == for comparison
      else if (op.find(">=") != std::string::npos)
         op = ">="; // Convert single = to == for comparison

      //std::cerr << "Debug - Extracted operator: '" << op << "'" << std::endl;

      auto filteredTable = applyFilter(currentTable, col, value, op);
      currentTable = filteredTable;
      return currentTable;
   }

   antlrcpp::Any visitFromClause(PipeQLParser::FromClauseContext* ctx) override {
      if (ctx->IDENTIFIER()) {
         return ctx->IDENTIFIER()->getText();
      }
      return nullptr;
   }

   antlrcpp::Any visitPipeOperator(PipeQLParser::PipeOperatorContext* ctx) override {
      if (ctx->whereOperator()) {
         return visitWhereOperator(ctx->whereOperator());
      }
      if (ctx->selectOperator()) {
         return visitSelectOperator(ctx->selectOperator());
      }
      return antlrcpp::Any();
   }

   antlrcpp::Any visitOrderByOperator(PipeQLParser::OrderByOperatorContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitUnionOperator(PipeQLParser::UnionOperatorContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitIntersectOperator(PipeQLParser::IntersectOperatorContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitExceptOperator(PipeQLParser::ExceptOperatorContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitAssertOperator(PipeQLParser::AssertOperatorContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitLimitClause(PipeQLParser::LimitClauseContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitOffsetClause(PipeQLParser::OffsetClauseContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitSelectExpression(PipeQLParser::SelectExpressionContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitOrderExpression(PipeQLParser::OrderExpressionContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitBooleanExpression(PipeQLParser::BooleanExpressionContext* ctx) override {
      if (ctx->comparisonOperator()) {
         auto op = visit(ctx->comparisonOperator());
         auto left = visit(ctx->expression(0));
         auto right = visit(ctx->expression(1));

         try {
            std::string opStr = std::any_cast<std::string>(op);
            std::string leftStr = std::any_cast<std::string>(left);
            std::string rightStr = std::any_cast<std::string>(right);
            return std::make_tuple(leftStr, rightStr, opStr);
         } catch (const std::bad_any_cast& e) {
            return nullptr;
         }
      }
      return nullptr;
   }

   antlrcpp::Any visitComparisonOperator(PipeQLParser::ComparisonOperatorContext* ctx) override {
      if (ctx->getText() == "=") return "==";
      if (ctx->getText() == "!=") return "!=";
      if (ctx->getText() == "<") return "<";
      if (ctx->getText() == "<=") return "<=";
      if (ctx->getText() == ">") return ">";
      if (ctx->getText() == ">=") return ">=";
      return "==";
   }

   antlrcpp::Any visitPayloadExpression(PipeQLParser::PayloadExpressionContext* ctx) override {
      return nullptr;
   }

   antlrcpp::Any visitExpression(PipeQLParser::ExpressionContext* ctx) override {
      if (ctx->IDENTIFIER()) {
         return ctx->IDENTIFIER()->getText();
      }
      if (ctx->literal()) {
         return visit(ctx->literal());
      }
      return nullptr;
   }

   antlrcpp::Any visitFunctionCall(PipeQLParser::FunctionCallContext* ctx) override {
      return antlrcpp::Any();
   }

   antlrcpp::Any visitLiteral(PipeQLParser::LiteralContext* ctx) override {
      if (ctx->STRING()) {
         std::string text = ctx->STRING()->getText();
         return text.substr(1, text.length() - 2);
      }
      if (ctx->NUMBER()) {
         return ctx->NUMBER()->getText();
      }
      return nullptr;
   }

   antlrcpp::Any visitAliasClause(PipeQLParser::AliasClauseContext* ctx) override {
      return antlrcpp::Any();
   }

   private:
   lingodb::catalog::Catalog& catalog;
   std::shared_ptr<arrow::Table> currentTable;
   std::string dbDir;

   std::shared_ptr<arrow::Table> loadTable(const std::string& tableName) {
      auto tableEntry = catalog.getTypedEntry<catalog::TableCatalogEntry>(tableName);
      if (!tableEntry) {
         std::cerr << "Table entry not found for: " << tableName << std::endl;
         return nullptr;
      }

      std::string arrowFile = dbDir + "/" + tableName + ".arrow";
      //   std::cerr << "Loading table from: " << arrowFile << std::endl;

      auto infile = arrow::io::ReadableFile::Open(arrowFile);
      if (!infile.ok()) {
         std::cerr << "Failed to open Arrow file: " << infile.status().ToString() << std::endl;
         return nullptr;
      }

      auto reader = arrow::ipc::RecordBatchFileReader::Open(*infile);
      if (!reader.ok()) {
         std::cerr << "Failed to open Arrow reader: " << reader.status().ToString() << std::endl;
         return nullptr;
      }

      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      for (int i = 0; i < (*reader)->num_record_batches(); i++) {
         auto batch = (*reader)->ReadRecordBatch(i);
         if (!batch.ok()) {
            std::cerr << "Failed to read batch " << i << ": " << batch.status().ToString() << std::endl;
            return nullptr;
         }
         batches.push_back(*batch);
      }

      auto table = arrow::Table::FromRecordBatches(batches);
      if (!table.ok()) {
         std::cerr << "Failed to create table from batches: " << table.status().ToString() << std::endl;
         return nullptr;
      }

      //   std::cerr << "Successfully loaded table with " << (*table)->num_rows() << " rows and "
      //             << (*table)->num_columns() << " columns" << std::endl;

      // Print schema for debugging
      //   std::cerr << "Table schema:" << std::endl;
      //   for (int i = 0; i < (*table)->num_columns(); i++) {
      //      std::cerr << "Column " << i << ": " << (*table)->schema()->field(i)->name()
      //               << " (" << (*table)->schema()->field(i)->type()->ToString() << ")" << std::endl;
      //   }

      return *table;
   }

   std::shared_ptr<arrow::Table> applyFilter(const std::shared_ptr<arrow::Table>& table,
                                             const std::string& column,
                                             const std::string& value,
                                             const std::string& op) {
      if (!table) {
         std::cerr << "Filter: Input table is null" << std::endl;
         return nullptr;
      }

      auto colIndex = table->schema()->GetFieldIndex(column);
      if (colIndex == -1) {
         std::cerr << "Filter: Column '" << column << "' not found in table" << std::endl;
         return nullptr;
      }

      std::vector<bool> filter;
      auto chunkedArray = std::static_pointer_cast<arrow::ChunkedArray>(table->column(colIndex));

      // Process each chunk
      for (int chunkIdx = 0; chunkIdx < chunkedArray->num_chunks(); chunkIdx++) {
         auto chunk = chunkedArray->chunk(chunkIdx);

         for (int64_t i = 0; i < chunk->length(); i++) {
            if (chunk->IsNull(i)) {
               filter.push_back(false);
               continue;
            }

            std::string rowValue;
            switch (chunk->type_id()) {
               case arrow::Type::INT8: {
                  auto array = std::static_pointer_cast<arrow::Int8Array>(chunk);
                  rowValue = std::to_string(array->Value(i));
                  break;
               }
               case arrow::Type::INT16: {
                  auto array = std::static_pointer_cast<arrow::Int16Array>(chunk);
                  rowValue = std::to_string(array->Value(i));
                  break;
               }
               case arrow::Type::INT32: {
                  auto array = std::static_pointer_cast<arrow::Int32Array>(chunk);
                  rowValue = std::to_string(array->Value(i));
                  break;
               }
               case arrow::Type::INT64: {
                  auto array = std::static_pointer_cast<arrow::Int64Array>(chunk);
                  rowValue = std::to_string(array->Value(i));
                  break;
               }
               case arrow::Type::FLOAT: {
                  auto array = std::static_pointer_cast<arrow::FloatArray>(chunk);
                  rowValue = std::to_string(array->Value(i));
                  break;
               }
               case arrow::Type::DOUBLE: {
                  auto array = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                  rowValue = std::to_string(array->Value(i));
                  break;
               }
               case arrow::Type::STRING: {
                  auto array = std::static_pointer_cast<arrow::StringArray>(chunk);
                  rowValue = array->GetString(i);
                  break;
               }
               case arrow::Type::BOOL: {
                  auto array = std::static_pointer_cast<arrow::BooleanArray>(chunk);
                  rowValue = array->Value(i) ? "true" : "false";
                  break;
               }
               default: {
                  auto scalar = chunk->GetScalar(i);
                  if (!scalar.ok()) {
                     std::cerr << "Error getting scalar value: " << scalar.status().ToString() << std::endl;
                     filter.push_back(false);
                     continue;
                  }
                  rowValue = scalar.ValueOrDie()->ToString();
                  break;
               }
            }

            bool matches = false;
            try {
               double num1 = std::stod(rowValue);
               double num2 = std::stod(value);

               if (op == "==")
                  matches = num1 == num2;
               else if (op == "!=")
                  matches = num1 != num2;
               else if (op == "<")
                  matches = num1 < num2;
               else if (op == "<=")
                  matches = num1 <= num2;
               else if (op == ">")
                  matches = num1 > num2;
               else if (op == ">=")
                  matches = num1 >= num2;
            } catch (...) {
               // If numeric conversion fails, fall back to string comparison
               if (op == "==")
                  matches = rowValue == value;
               else if (op == "!=")
                  matches = rowValue != value;
               else if (op == "<")
                  matches = rowValue < value;
               else if (op == "<=")
                  matches = rowValue <= value;
               else if (op == ">")
                  matches = rowValue > value;
               else if (op == ">=")
                  matches = rowValue >= value;
            }
            filter.push_back(matches);
         }
      }

      //std::cerr << "Filter array size: " << filter.size() << std::endl;
      //std::cerr << "Number of true values in filter: " << std::count(filter.begin(), filter.end(), true) << std::endl;

      arrow::BooleanBuilder builder;
      if (!builder.AppendValues(filter).ok()) {
         std::cerr << "Error appending filter values" << std::endl;
         return nullptr;
      }
      std::shared_ptr<arrow::BooleanArray> filterArray;
      if (!builder.Finish(&filterArray).ok()) {
         std::cerr << "Error finalizing filter array" << std::endl;
         return nullptr;
      }

      auto options = arrow::compute::FilterOptions::Defaults();
      auto result = arrow::compute::Filter(table, filterArray, options);
      if (!result.ok()) {
         std::cerr << "Error applying filter: " << result.status().ToString() << std::endl;
         return nullptr;
      }

      auto filteredTable = result.ValueOrDie().table();
      return filteredTable;
   }
};
} // namespace interpreter::pipeql

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
} // anonymous namespace

int main(int argc, char** argv) {
   if (argc != 3) {
      std::cerr << "Usage: " << argv[0] << " <db_dir> <pipeql_query>" << std::endl;
      std::cerr << "Example: " << argv[0] << " /path/to/db 'FROM users | SELECT name, age | WHERE age > 18'" << std::endl;
      return 1;
   }

   std::string dbDir = argv[1];
   std::string query = argv[2];

   try {
      auto session = runtime::Session::createSession(dbDir, true);
      auto catalog = session->getCatalog();

      // Create ANTLR input stream from query
      antlr4::ANTLRInputStream input(query);
      PipeQLLexer lexer(&input);
      antlr4::CommonTokenStream tokens(&lexer);
      PipeQLParser parser(&tokens);

      // Parse the query
      auto* ast = parser.query();
      if (parser.getNumberOfSyntaxErrors() > 0) {
         std::cerr << "Error: Invalid PipeQL query syntax" << std::endl;
         return 1;
      }

      // Create visitor and execute query
      interpreter::pipeql::InterpreterPipeQLVisitor visitor(*catalog, dbDir);
      
      auto start = std::chrono::high_resolution_clock::now();
      auto result = visitor.visitQuery(ast);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cerr << "Query execution time: " << duration.count() << " microseconds" << std::endl;

      try {
         auto table = std::any_cast<std::shared_ptr<arrow::Table>>(result);
         if (table) {
            printTable(table);
         } else {
            std::cerr << "Error: Query execution failed" << std::endl;
            return 1;
         }
      } catch (const std::bad_any_cast& e) {
         std::cerr << "Error: Failed to process query result" << std::endl;
         return 1;
      }

   } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
   }

   return 0;
}