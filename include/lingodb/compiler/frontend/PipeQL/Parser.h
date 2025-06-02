#ifndef LINGODB_COMPILER_FRONTEND_PIPEQL_PARSER_H
#define LINGODB_COMPILER_FRONTEND_PIPEQL_PARSER_H
#include "PipeQL/src/include/pipeql/parser/PipeQLLexer.h"
#include "PipeQL/src/include/pipeql/parser/PipeQLParser.h"
#include "antlr4-runtime.h"

#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ScopedHashTable.h"

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <variant>

namespace lingodb::compiler::dialect::tuples {
struct Column; // Changed from class to struct
} // namespace lingodb::compiler::dialect::tuples

namespace lingodb::compiler::frontend::pipeql {

// Forward declarations
struct Node;
struct CreateStmt;
struct InsertStmt;
struct VariableSetStmt;
struct CopyStmt;
struct SelectStmt;
struct List;
enum class SortByDir;

struct StringInfo {
   static bool isEqual(std::string a, std::string b) { return a == b; }
   static std::string getEmptyKey() { return ""; }
   static std::string getTombstoneKey() { return "-"; }
   static size_t getHashValue(std::string str) { return std::hash<std::string>{}(str); }
};

struct TranslationContext {
   std::stack<mlir::Value> currTuple;
   std::unordered_set<const dialect::tuples::Column*> useZeroInsteadNull;
   std::stack<std::vector<std::pair<std::string, const dialect::tuples::Column*>>> definedAttributes;

   llvm::ScopedHashTable<std::string, const dialect::tuples::Column*, StringInfo> resolver;
   using ResolverScope = llvm::ScopedHashTable<std::string, const dialect::tuples::Column*, StringInfo>::ScopeTy;
   struct TupleScope {
      TranslationContext* context;
      bool active;
      TupleScope(TranslationContext* context) : context(context) {
         context->currTuple.push(context->currTuple.top());
      }

      ~TupleScope() {
         context->currTuple.pop();
      }
   };

   TranslationContext() : currTuple(), resolver() {
      currTuple.push(mlir::Value());
      definedAttributes.push({});
   }
   mlir::Value getCurrentTuple() {
      return currTuple.top();
   }
   void setCurrentTuple(mlir::Value v) {
      currTuple.top() = v;
   }
   void mapAttribute(ResolverScope& scope, std::string name, const dialect::tuples::Column* attr) {
      definedAttributes.top().push_back({name, attr});
      resolver.insertIntoScope(&scope, name, attr);
   }
   const dialect::tuples::Column* getAttribute(std::string name) {
      const auto* res = resolver.lookup(name);
      if (!res) {
         throw std::runtime_error("could not resolve '" + name + "'");
      }
      return res;
   }
   TupleScope createTupleScope() {
      return TupleScope(this);
   }
   ResolverScope createResolverScope() {
      return ResolverScope(resolver);
   }
   struct DefineScope {
      TranslationContext& context;
      DefineScope(TranslationContext& context) : context(context) {
         context.definedAttributes.push({});
      }
      ~DefineScope() {
         context.definedAttributes.pop();
      }
   };
   DefineScope createDefineScope() {
      return DefineScope(*this);
   }
   const std::vector<std::pair<std::string, const dialect::tuples::Column*>>& getAllDefinedColumns() {
      return definedAttributes.top();
   }
   void removeFromDefinedColumns(const dialect::tuples::Column* col) {
      auto& currDefinedColumns = definedAttributes.top();
      auto start = currDefinedColumns.begin();
      auto end = currDefinedColumns.end();
      auto position = std::find_if(start, end, [&](auto el) { return el.second == col; });
      if (position != currDefinedColumns.end()) {
         currDefinedColumns.erase(position);
      }
   }

   void replace(ResolverScope& scope, const dialect::tuples::Column* col, const dialect::tuples::Column* col2) {
      auto& currDefinedColumns = definedAttributes.top();
      auto start = currDefinedColumns.begin();
      auto end = currDefinedColumns.end();
      std::vector<std::string> toReplace;
      while (start != end) {
         auto position = std::find_if(start, end, [&](auto el) { return el.second == col; });
         if (position != currDefinedColumns.end()) {
            start = position + 1;
            toReplace.push_back(position->first);
         } else {
            start = end;
         }
      }
      for (auto s : toReplace) {
         mapAttribute(scope, s, col2);
      }
   }
};

class TargetInfo {
   public:
   TargetInfo() = default;
   ~TargetInfo() = default;

   void map(std::string name, const dialect::tuples::Column* attr) {
      namedResults.push_back({name, attr});
   }

   const std::vector<std::pair<std::string, const dialect::tuples::Column*>>& getNamedResults() const {
      return namedResults;
   }

   void debugPrint() const {
      std::cout << "TargetInfo contents:" << std::endl;
      for (const auto& [name, attr] : namedResults) {
         std::cout << "  - Name: " << name << std::endl;
      }
   }

   void addNamedResult(const std::string& name, const dialect::tuples::Column* column) {
      namedResults.push_back({name, column});
   }

   private:
   std::vector<std::pair<std::string, const dialect::tuples::Column*>> namedResults;
};

class Parser {
   public:
   Parser(std::string pipeql, lingodb::catalog::Catalog& catalog, mlir::ModuleOp moduleOp)
      : attrManager(moduleOp.getContext()->getLoadedDialect<dialect::tuples::TupleStreamDialect>()->getColumnManager()), pipeql(std::move(pipeql)), catalog(catalog), moduleOp(moduleOp), parallelismAllowed(true), targetInfo() {}

   ~Parser() = default;

   // Core translation methods
   std::optional<mlir::Value> translate(mlir::OpBuilder& builder);

   private:
   dialect::tuples::ColumnManager& attrManager;
   std::string pipeql;
   lingodb::catalog::Catalog& catalog;
   mlir::ModuleOp moduleOp;
   bool parallelismAllowed;
   TargetInfo targetInfo;
   std::string currentTable;
};

} // namespace lingodb::compiler::frontend::pipeql

#endif // LINGODB_COMPILER_FRONTEND_PIPEQL_PARSER_H