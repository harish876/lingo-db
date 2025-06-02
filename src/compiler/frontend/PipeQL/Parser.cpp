#include "lingodb/compiler/frontend/PipeQL/Parser.h"
#include "MLIRPipeQLVisitor.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/runtime/ExecutionContext.h"
#include "lingodb/compiler/runtime/RelationHelper.h"
#include "lingodb/utility/Serialization.h"
#include <iostream>

namespace lingodb::compiler::frontend::pipeql {

std::optional<mlir::Value> Parser::translate(mlir::OpBuilder& builder) {
   antlr4::ANTLRInputStream input(pipeql);

   PipeQLLexer lexer(&input);
   antlr4::CommonTokenStream tokens(&lexer);
   PipeQLParser parser(&tokens);

   auto* ast = parser.query();

   MLIRPipeQLVisitor visitor(catalog, this->attrManager, this->targetInfo);

   mlir::Block* block = new mlir::Block;
   mlir::Type localTableType;
   TranslationContext context;
   auto scope = context.createResolverScope();

   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(block);

      auto [tree, newTargetInfo] = visitor.translateSelectStmt(builder, ast, context, scope);
      if (!tree) {
         std::cerr << "Error: MLIR tree not created yet" << std::endl;
         return std::nullopt;
      }

      targetInfo = newTargetInfo;
      std::vector<mlir::Attribute> attrs;
      std::vector<mlir::Attribute> names;
      std::vector<mlir::Attribute> colMemberNames;
      std::vector<mlir::Attribute> colTypes;

      auto& memberManager = builder.getContext()->getLoadedDialect<dialect::subop::SubOperatorDialect>()->getMemberManager();
      for (const auto& x : newTargetInfo.getNamedResults()) {
         if (x.first == "primaryKeyHashValue") continue;
         names.push_back(builder.getStringAttr(x.first));
         auto colMemberName = memberManager.getUniqueMember(x.first.empty() ? "unnamed" : x.first);
         auto columnType = x.second->type;
         attrs.push_back(attrManager.createRef(x.second));
         colTypes.push_back(mlir::TypeAttr::get(columnType));
         colMemberNames.push_back(builder.getStringAttr(colMemberName));
      }

      localTableType = dialect::subop::LocalTableType::get(
         builder.getContext(),
         dialect::subop::StateMembersAttr::get(builder.getContext(),
                                               builder.getArrayAttr(colMemberNames),
                                               builder.getArrayAttr(colTypes)),
         builder.getArrayAttr(names));

      mlir::Value result = builder.create<dialect::relalg::MaterializeOp>(
         builder.getUnknownLoc(),
         localTableType,
         tree,
         builder.getArrayAttr(attrs),
         builder.getArrayAttr(names));

      builder.create<dialect::relalg::QueryReturnOp>(builder.getUnknownLoc(), result);
   }

   dialect::relalg::QueryOp queryOp = builder.create<dialect::relalg::QueryOp>(
      builder.getUnknownLoc(),
      mlir::TypeRange{localTableType},
      mlir::ValueRange{});

   queryOp.getQueryOps().getBlocks().clear();
   queryOp.getQueryOps().push_back(block);
   return queryOp.getResults()[0];
}

} // namespace lingodb::compiler::frontend::pipeql
