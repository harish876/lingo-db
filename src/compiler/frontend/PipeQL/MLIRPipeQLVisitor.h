#pragma once

#include "PipeQL/src/include/pipeql/parser/PipeQLBaseVisitor.h"
#include "PipeQL/src/include/pipeql/parser/PipeQLParser.h"
#include "lingodb/catalog/Catalog.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include "lingodb/compiler/frontend/PipeQL/Parser.h"
#include <string>

namespace lingodb::compiler::frontend::pipeql {

class MLIRPipeQLVisitor : public PipeQLBaseVisitor {
   public:
   MLIRPipeQLVisitor(lingodb::catalog::Catalog& catalog,
                     lingodb::compiler::dialect::tuples::ColumnManager& attrManager,
                     lingodb::compiler::frontend::pipeql::TargetInfo& targetInfo)
      : catalog(catalog), attrManager(attrManager), targetInfo(targetInfo) {}

   void debugPrint();
   std::pair<mlir::Value, TargetInfo> translateSelectStmt(mlir::OpBuilder& builder, PipeQLParser::QueryContext* ctx, TranslationContext& context, TranslationContext::ResolverScope& scope);

   antlrcpp::Any visitQuery(PipeQLParser::QueryContext* ctx) override;
   antlrcpp::Any visitSelectOperator(PipeQLParser::SelectOperatorContext* ctx) override;
   antlrcpp::Any visitWhereOperator(PipeQLParser::WhereOperatorContext* ctx) override;
   antlrcpp::Any visitFromClause(PipeQLParser::FromClauseContext* ctx) override;
   antlrcpp::Any visitPipeOperator(PipeQLParser::PipeOperatorContext* ctx) override;
   antlrcpp::Any visitOrderByOperator(PipeQLParser::OrderByOperatorContext* ctx) override;
   antlrcpp::Any visitUnionOperator(PipeQLParser::UnionOperatorContext* ctx) override;
   antlrcpp::Any visitIntersectOperator(PipeQLParser::IntersectOperatorContext* ctx) override;
   antlrcpp::Any visitExceptOperator(PipeQLParser::ExceptOperatorContext* ctx) override;
   antlrcpp::Any visitAssertOperator(PipeQLParser::AssertOperatorContext* ctx) override;
   antlrcpp::Any visitLimitClause(PipeQLParser::LimitClauseContext* ctx) override;
   antlrcpp::Any visitOffsetClause(PipeQLParser::OffsetClauseContext* ctx) override;
   antlrcpp::Any visitSelectExpression(PipeQLParser::SelectExpressionContext* ctx) override;
   antlrcpp::Any visitOrderExpression(PipeQLParser::OrderExpressionContext* ctx) override;
   antlrcpp::Any visitBooleanExpression(PipeQLParser::BooleanExpressionContext* ctx) override;
   antlrcpp::Any visitComparisonOperator(PipeQLParser::ComparisonOperatorContext* ctx) override;
   antlrcpp::Any visitPayloadExpression(PipeQLParser::PayloadExpressionContext* ctx) override;
   antlrcpp::Any visitExpression(PipeQLParser::ExpressionContext* ctx) override;
   antlrcpp::Any visitFunctionCall(PipeQLParser::FunctionCallContext* ctx) override;
   antlrcpp::Any visitLiteral(PipeQLParser::LiteralContext* ctx) override;
   antlrcpp::Any visitAliasClause(PipeQLParser::AliasClauseContext* ctx) override;

   mlir::Type createBaseTypeFromColumnType(mlir::MLIRContext* context, const lingodb::catalog::Type& t);
    //   return t.getMLIRTypeCreator()->createType(context);

   mlir::Type createTypeForColumn(mlir::MLIRContext* context, const lingodb::catalog::Column& colDef);
    //   mlir::Type baseType = createBaseTypeFromColumnType(context, colDef.getLogicalType());
    //   return colDef.getIsNullable() ? db::NullableType::get(context, baseType) : baseType;

   private:
   enum class SelectType {
      ALL,
      COLUMNS,
      EMPTY
   };

   SelectType getSelectType(PipeQLParser::SelectOperatorContext* ctx) {
      if (ctx->selectExpression().empty()) {
         return SelectType::EMPTY;
      }
      if (ctx->selectExpression().size() == 1 &&
          ctx->selectExpression(0)->getText() == "*") {
         return SelectType::ALL;
      }
      return SelectType::COLUMNS;
   }

   lingodb::catalog::Catalog& catalog;
   lingodb::compiler::dialect::tuples::ColumnManager& attrManager;
   lingodb::compiler::frontend::pipeql::TargetInfo& targetInfo;
   std::string currentTable;
};

} // namespace lingodb::compiler::frontend::pipeql