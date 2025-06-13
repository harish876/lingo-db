#ifndef LINGODB_EXECUTION_TIMING_H
#define LINGODB_EXECUTION_TIMING_H
#include "Error.h"

#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>
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
      double total = 0.0;
      for (auto [name, t] : timing) {
         total += t;
      }
      timing["total"] = total;
      std::vector<std::string> printOrder = {"QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", "executionTime", "total"};
      
      // Print table header
      std::cout << std::endl << std::endl;
      std::cout << std::string(printOrder.size() * 15 + 12, '-') << std::endl;
      std::cout << "| " << std::setw(10) << std::left << "name" << " |";
      for (auto n : printOrder) {
         std::cout << std::setw(13) << std::left << n << " |";
      }
      std::cout << std::endl;
      std::cout << std::string(printOrder.size() * 15 + 12, '-') << std::endl;
      
      // Print data row
      std::cout << "| " << std::setw(10) << std::left << queryName << " |";
      for (auto n : printOrder) {
         if (timing.contains(n)) {
            std::cout << std::setw(13) << std::left << timing[n] << " |";
         } else {
            std::cout << std::setw(13) << std::left << "" << " |";
         }
      }
      std::cout << std::endl;
      std::cout << std::string(printOrder.size() * 15 + 12, '-') << std::endl;
   }
};
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_TIMING_H