#include "MLIRPipeQLVisitor.h"
#include "lingodb/catalog/MLIRTypes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/frontend/SQL/Parser.h"
#include <iostream>

namespace lingodb::compiler::frontend::pipeql {

mlir::Type MLIRPipeQLVisitor::createBaseTypeFromColumnType(mlir::MLIRContext* context, const lingodb::catalog::Type& t) {
   return t.getMLIRTypeCreator()->createType(context);
}

mlir::Type MLIRPipeQLVisitor::createTypeForColumn(mlir::MLIRContext* context, const lingodb::catalog::Column& colDef) {
   mlir::Type baseType = createBaseTypeFromColumnType(context, colDef.getLogicalType());
   return colDef.getIsNullable() ? dialect::db::NullableType::get(context, baseType) : baseType;
}

void MLIRPipeQLVisitor::debugPrint() {
   std::cout << "\n==== Debugging MLIRPipeQLVisitor ====" << std::endl;
   std::cout << "Current table: " << currentTable << std::endl;

   std::cout << "\nTargetInfo:" << std::endl;
   targetInfo.debugPrint();

   // Debug print MLIR array construction
   std::cout << "\nMLIR Array Construction:" << std::endl;
   for (const auto& x : targetInfo.getNamedResults()) {
      if (x.first == "primaryKeyHashValue") continue;
      std::cout << "Processing column: " << x.first << std::endl;
      std::cout << "  - Name: " << x.first << std::endl;
      std::cout << "  - Column Type: ";
      llvm::raw_ostream& os = llvm::outs();
      x.second->type.print(os);
      os << "\n";
   }

   std::cout << "==== End Debugging MLIRPipeQLVisitor ====\n"
             << std::endl;
}

std::pair<mlir::Value, TargetInfo> MLIRPipeQLVisitor::translateSelectStmt(mlir::OpBuilder& builder, PipeQLParser::QueryContext* ctx, TranslationContext& context, TranslationContext::ResolverScope& scope) {
   std::string tableName;
   mlir::Value tree;
   if (ctx->fromClause()) {
      try {
         tableName = std::any_cast<std::string>(visitFromClause(ctx->fromClause()));
         currentTable = tableName;
      } catch (const std::bad_any_cast& e) {
         std::cerr << "Error: Failed to get table name from FROM clause" << std::endl;
         std::cerr << "Details: " << e.what() << std::endl;
         return std::make_pair(tree, targetInfo);
      }
   }

   for (auto* stage : ctx->pipeOperator()) {
      if (stage->selectOperator()) {
         mlir::Value resultTree = translateSelectOperator(builder, stage->selectOperator(), context, scope);
         tree = resultTree;
      }
      if (stage->whereOperator()) {
         mlir::Value resultTree  = translateWhereOperator(builder, tree, stage->whereOperator(), context, scope);
         tree = resultTree;
      }
   }
   return std::make_pair(tree, targetInfo);
}

mlir::Value MLIRPipeQLVisitor::translateWhereOperator(mlir::OpBuilder& builder, mlir::Value tree, PipeQLParser::WhereOperatorContext* ctx, TranslationContext& context, TranslationContext::ResolverScope& scope) {
   mlir::Block* pred = translatePredicate(builder, ctx, context);
   auto sel = builder.create<dialect::relalg::SelectionOp>(
      builder.getUnknownLoc(),
      dialect::tuples::TupleStreamType::get(builder.getContext()),
      tree
   );
   sel.getPredicate().push_back(pred);
   tree = sel.getResult();
   return tree;
}

mlir::Block* MLIRPipeQLVisitor::translatePredicate(mlir::OpBuilder& builder, PipeQLParser::WhereOperatorContext* ctx, TranslationContext& context) {
   auto* block = new mlir::Block;
   mlir::OpBuilder predBuilder(builder.getContext());
   block->addArgument(dialect::tuples::TupleType::get(builder.getContext()), builder.getUnknownLoc());
   auto tupleScope = context.createTupleScope();
   context.setCurrentTuple(block->getArgument(0));

   predBuilder.setInsertionPointToStart(block);
   mlir::Value expr = translateExpression(predBuilder, ctx, context);
   predBuilder.create<dialect::tuples::ReturnOp>(builder.getUnknownLoc(), expr);
   return block;
}

mlir::Value MLIRPipeQLVisitor::translateExpression(mlir::OpBuilder& builder, PipeQLParser::WhereOperatorContext* ctx, TranslationContext& context) {
   auto loc = builder.getUnknownLoc();
   auto* booleanExpr = ctx->booleanExpression();
   std::string whereExpression = booleanExpr->getText();
   
   dialect::db::DBCmpPredicate pred;
   if (whereExpression.find("==") != std::string::npos) {
      pred = dialect::db::DBCmpPredicate::eq;
   } else if (whereExpression.find("!=") != std::string::npos) {
      pred = dialect::db::DBCmpPredicate::neq;
   } else if (whereExpression.find("<=") != std::string::npos) {
      pred = dialect::db::DBCmpPredicate::lte;
   } else if (whereExpression.find(">=") != std::string::npos) {
      pred = dialect::db::DBCmpPredicate::gte;
   } else if (whereExpression.find("<") != std::string::npos) {
      pred = dialect::db::DBCmpPredicate::lt;
   } else if (whereExpression.find(">") != std::string::npos) {
      pred = dialect::db::DBCmpPredicate::gt;
   } else {
      return mlir::Value();
   }

   auto* columnExpr = booleanExpr->expression(0); // Column name
   auto* expr = booleanExpr->expression(1); // Value to compare

   std::string attrName = columnExpr->getText();
   int exprValue = std::stoi(expr->getText()); // handle exception here

   mlir::Value left, right;
   const auto* attr = context.getAttribute(attrName);
   left = builder.create<dialect::tuples::GetColumnOp>(builder.getUnknownLoc(), attr->type, attrManager.createRef(attr), context.getCurrentTuple());
   right = builder.create<dialect::db::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(exprValue));
   auto ct = sql::SQLTypeInference::toCommonBaseTypes(builder, {left, right});
   return builder.create<dialect::db::CmpOp>(
      builder.getUnknownLoc(),
      pred,
      ct[0],
      ct[1]);
}

mlir::Value MLIRPipeQLVisitor::translateSelectOperator(mlir::OpBuilder& builder, PipeQLParser::SelectOperatorContext* ctx, TranslationContext& context, TranslationContext::ResolverScope& scope) {
   visitSelectOperator(ctx);

   mlir::Value tree;
   auto maybeRel = catalog.getTypedEntry<lingodb::catalog::TableCatalogEntry>(currentTable);
   std::string alias = currentTable;
   auto rel = maybeRel.value();

   //TODO: check why unique scope gives user_u_1
   // char lastCharacter = alias.back();
   // std::string scopeName = attrManager.getUniqueScope(alias + (isdigit(lastCharacter) ? "_" : ""));

   std::vector<mlir::NamedAttribute> columns;
   for (auto c : rel->getColumns()) {
      auto attrDef = attrManager.createDef(currentTable, c.getColumnName());
      attrDef.getColumn().type = createTypeForColumn(builder.getContext(), c);
      columns.push_back(builder.getNamedAttr(c.getColumnName(), attrDef));
      context.mapAttribute(scope, c.getColumnName(), &attrDef.getColumn()); //todo check for existing and overwrite...
      context.mapAttribute(scope, alias + "." + c.getColumnName(), &attrDef.getColumn());
   }
   
   tree = builder.create<dialect::relalg::BaseTableOp>(
      builder.getUnknownLoc(),
      dialect::tuples::TupleStreamType::get(builder.getContext()),
      currentTable,
      builder.getDictionaryAttr(columns));

   return tree;
}

antlrcpp::Any MLIRPipeQLVisitor::visitSelectOperator(PipeQLParser::SelectOperatorContext* ctx) {
   switch (getSelectType(ctx)) {
      case SelectType::ALL: {
         auto tableEntry = catalog.getTypedEntry<lingodb::catalog::TableCatalogEntry>(currentTable);
         if (!tableEntry) {
            std::cerr << "Error: Table " << currentTable << " not found in catalog" << std::endl;
            return antlrcpp::Any();
         }

         auto columnNames = tableEntry.value()->getColumnNames();
         std::unordered_set<const dialect::tuples::Column*> handledAttrs;

         for (const auto& columnName : columnNames) {
            auto attrDef = attrManager.createDef(currentTable, columnName);
            if (!handledAttrs.contains(&attrDef.getColumn())) {
               targetInfo.addNamedResult(columnName, &attrDef.getColumn());
               handledAttrs.insert(&attrDef.getColumn());
            }
         }
         break;
      }
      case SelectType::COLUMNS: {
         std::unordered_set<const dialect::tuples::Column*> handledAttrs;
         for (auto* selectExpr : ctx->selectExpression()) {
            std::string column;
            if (selectExpr->expression()) {
               column = selectExpr->expression()->getText();
            } else {
               column = selectExpr->getText();
            }

            auto attrDef = attrManager.createDef(currentTable, column);
            if (!handledAttrs.contains(&attrDef.getColumn())) {
               targetInfo.addNamedResult(column, &attrDef.getColumn());
               handledAttrs.insert(&attrDef.getColumn());
            }
         }
         break;
      }
      case SelectType::EMPTY:
         std::cerr << "Warning: Empty select expression list" << std::endl;
         break;
   }
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitWhereOperator(PipeQLParser::WhereOperatorContext* ctx) {
   auto* booleanExpr = ctx->booleanExpression();
   std::string whereExpression = booleanExpr->getText();

   if (whereExpression.find("==") != std::string::npos) {
      auto* columnExpr = booleanExpr->expression(0); // Column name
      auto* expr = booleanExpr->expression(1); // Upper bound

      std::string column = columnExpr->getText();
      int exprValue = std::stoi(expr->getText());
      std::cout << "Column Name: " << column << " Expression Value: " << exprValue
                << std::endl;
   }
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitFromClause(PipeQLParser::FromClauseContext* ctx) {
   if (!ctx->IDENTIFIER()) {
      std::cerr << "Error: FROM clause missing table identifier" << std::endl;
      return antlrcpp::Any();
   }

   std::string tableName = ctx->IDENTIFIER()->getText();
   if (tableName.empty()) {
      std::cerr << "Error: Empty table name in FROM clause" << std::endl;
      return antlrcpp::Any();
   }

   currentTable = tableName;
   return tableName;
}
//NOT USING THIS AS OF NOW
antlrcpp::Any MLIRPipeQLVisitor::visitPipeOperator(PipeQLParser::PipeOperatorContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitOrderByOperator(PipeQLParser::OrderByOperatorContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitUnionOperator(PipeQLParser::UnionOperatorContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitIntersectOperator(PipeQLParser::IntersectOperatorContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitExceptOperator(PipeQLParser::ExceptOperatorContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitAssertOperator(PipeQLParser::AssertOperatorContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitLimitClause(PipeQLParser::LimitClauseContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitOffsetClause(PipeQLParser::OffsetClauseContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitSelectExpression(PipeQLParser::SelectExpressionContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitOrderExpression(PipeQLParser::OrderExpressionContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitBooleanExpression(PipeQLParser::BooleanExpressionContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitComparisonOperator(PipeQLParser::ComparisonOperatorContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitPayloadExpression(PipeQLParser::PayloadExpressionContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitExpression(PipeQLParser::ExpressionContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitFunctionCall(PipeQLParser::FunctionCallContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitLiteral(PipeQLParser::LiteralContext* ctx) {
   return antlrcpp::Any();
}

antlrcpp::Any MLIRPipeQLVisitor::visitAliasClause(PipeQLParser::AliasClauseContext* ctx) {
   return antlrcpp::Any();
}
//TODO: refactor around this
antlrcpp::Any MLIRPipeQLVisitor::visitQuery(PipeQLParser::QueryContext* ctx) {
   std::string tableName;
   if (ctx->fromClause()) {
      try {
         tableName = std::any_cast<std::string>(visitFromClause(ctx->fromClause()));
      } catch (const std::bad_any_cast& e) {
         std::cerr << "Error: Failed to get table name from FROM clause" << std::endl;
         std::cerr << "Details: " << e.what() << std::endl;
         return antlrcpp::Any();
      }
   }

   for (auto* stage : ctx->pipeOperator()) {
      visitPipeOperator(stage);
   }

   return antlrcpp::Any();
}

} // namespace lingodb::compiler::frontend::pipeql
