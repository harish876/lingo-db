#ifndef LINGODB_EXECUTION_TIMING_H
#define LINGODB_EXECUTION_TIMING_H
#include "Error.h"

#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

namespace lingodb::execution {
class TimingProcessor {
   public:
   virtual void addTiming(const std::unordered_map<std::string, double>& timing) = 0;
   virtual void process() = 0;
   virtual ~TimingProcessor() {}
};

class TimingPrinter : public TimingProcessor {
   std::unordered_map<std::string, double> timing;
   std::string queryName;

   public:
   TimingPrinter(std::string queryFile) {
      if (queryFile.find('/') != std::string::npos) {
         queryName = queryFile.substr(queryFile.find_last_of("/\\") + 1);
      } else {
         queryName = queryFile;
      }
   }

   void addTiming(const std::unordered_map<std::string, double>& timing) override {
      this->timing.insert(timing.begin(), timing.end());
   }

   void process() override {
      // Calculate total excluding the executionTime
      double total = 0.0;
      for (auto [name, t] : timing) {
         if (name != "executionTime") {
            total += t;
         }
      }
      // Add executionTime to total
      if (timing.contains("executionTime")) {
         total += timing["executionTime"];
      }
      timing["total"] = total;

      // Define column order and widths
      std::vector<std::string> printOrder = {
         "QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", 
         "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", 
         "executionTime", "total"
      };
      
      // Calculate column widths
      std::unordered_map<std::string, size_t> colWidths;
      colWidths["name"] = std::max(queryName.length(), size_t(20));
      
      for (const auto& col : printOrder) {
         colWidths[col] = col.length();
         if (timing.contains(col)) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << timing[col];
            colWidths[col] = std::max(colWidths[col], ss.str().length());
         }
      }

      // Print header
      std::cout << "\n+" << std::string(colWidths["name"], '-') << "+";
      for (size_t i = 0; i < printOrder.size(); ++i) {
         std::cout << std::string(colWidths[printOrder[i]], '-');
         if (i < printOrder.size() - 1) std::cout << "+";
      }
      std::cout << "+\n";

      // Print column names
      std::cout << "|" << std::setw(colWidths["name"]) << "name" << "|";
      for (const auto& col : printOrder) {
         std::cout << std::setw(colWidths[col]) << col << "|";
      }
      std::cout << "\n";

      // Print separator
      std::cout << "+" << std::string(colWidths["name"], '-') << "+";
      for (size_t i = 0; i < printOrder.size(); ++i) {
         std::cout << std::string(colWidths[printOrder[i]], '-');
         if (i < printOrder.size() - 1) std::cout << "+";
      }
      std::cout << "+\n";

      // Print data row
      std::cout << "|" << std::setw(colWidths["name"]) << queryName << "|";
      for (const auto& col : printOrder) {
         if (timing.contains(col)) {
            std::cout << std::fixed << std::setprecision(3) << std::setw(colWidths[col]) << timing[col] << "|";
         } else {
            std::cout << std::setw(colWidths[col]) << "" << "|";
         }
      }
      std::cout << "\n";

      // Print footer
      std::cout << "+" << std::string(colWidths["name"], '-') << "+";
      for (size_t i = 0; i < printOrder.size(); ++i) {
         std::cout << std::string(colWidths[printOrder[i]], '-');
         if (i < printOrder.size() - 1) std::cout << "+";
      }
      std::cout << "+\n\n";
   }
};
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_TIMING_H

