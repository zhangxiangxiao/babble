// Combine the count of grams.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "cppglob/glob.hpp"

DEFINE_string(input_prefix, "data/tinyshakespeare/train_gram_count_sorted.",
              "Input file prefix.");
DEFINE_string(output, "data/tinyshakespeare/train_gram_count_combined.txt",
              "Output file.");

namespace fs = std::filesystem;
typedef std::pair<std::string, int64_t> KeyValuePair;
typedef std::tuple<std::string, int64_t, size_t> ItemTuple;
typedef std::priority_queue<
  ItemTuple, std::vector<ItemTuple>, std::greater<ItemTuple>> PriorityQueue;


bool readKeyValue(FILE *fd, KeyValuePair *kv) {
  static char buffer[1024 + 1];
  kv->first.clear();
  // Skip the beginning white spaces and read the first hex.
  if (fscanf(fd, " %1024[0123456789ABCDEFabcdef]", buffer) != 1) {
    return false;
  }
  kv->first.append(buffer);
  while (fscanf(fd, "%1024[0123456789ABCDEFabcdef]", buffer) == 1) {
    kv->first.append(buffer);
  }
  // Skip the beginning white spaces and read the first value.
  if (fscanf(fd, " %lu", &kv->second) != 1) {
    return false;
  }
  return true;
}

void writeKeyValue(const KeyValuePair &kv, FILE *fd) {
  fprintf(fd, "%s %lu\n", kv.first.c_str(), kv.second);
}

int combineGramCount() {
  LOG(INFO) << "Load input from prefix " << FLAGS_input_prefix << ".";
  std::vector<fs::path> input_files = cppglob::glob(FLAGS_input_prefix + "*");
  std::vector<FILE *> input_fds;
  for (const fs::path& input_file : input_files) {
    LOG(INFO) << "Open file " << input_file.c_str() << ".";
    input_fds.push_back(fopen(input_file.c_str(), "r"));
  }
  // Initialize input keys and values.
  PriorityQueue pqueue;
  KeyValuePair input_kv;
  for (int64_t i = 0; i < input_fds.size(); ++i) {
    FILE *input_fd = input_fds[i];
    if (readKeyValue(input_fd, &input_kv)) {
      pqueue.push(ItemTuple(input_kv.first, input_kv.second, i));
    } else {
      LOG(INFO) << "Close file " << input_files[i].c_str() << ".";
      fclose(input_fd);
    }
  }
  LOG(INFO) << "Write output to " << FLAGS_output << ".";
  FILE *output_fd = fopen(FLAGS_output.c_str(), "w");
  int64_t processed_keys = 0;
  KeyValuePair output_kv;
  ItemTuple input_item;
  while (!pqueue.empty()) {
    input_item = pqueue.top();
    pqueue.pop();
    FILE *input_fd = input_fds[std::get<2>(input_item)];
    if (readKeyValue(input_fd, &input_kv)) {
      pqueue.push(ItemTuple(
          input_kv.first, input_kv.second, std::get<2>(input_item)));
    } else {
      LOG(INFO) << "Close file "
                << input_files[std::get<2>(input_item)].c_str() << ".";
      fclose(input_fd);
    }
    if (output_kv.first == std::get<0>(input_item)) {
      output_kv.second = output_kv.second + std::get<1>(input_item);
    } else {
      if (!output_kv.first.empty()) {
        if (processed_keys % 100000 == 0) {
          LOG(INFO) << "Processed keys " << processed_keys << ".";
        }
        writeKeyValue(output_kv, output_fd);
        processed_keys = processed_keys + 1;
      }
      output_kv.first = std::get<0>(input_item);
      output_kv.second = std::get<1>(input_item);
    }
  }
  // Write last output key and value.
  if (!output_kv.first.empty()) {
    writeKeyValue(output_kv, output_fd);
    processed_keys = processed_keys + 1;
  }
  fclose(output_fd);
  LOG(INFO) << "Processed keys " << processed_keys << ".";
  return 0;
}

int main(int argc, char *argv[]) {
  // Change default arguments.
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  // Initialize Google gflags to parse command-line arguments.
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Initialize Google glogs for logging.
  ::google::InitGoogleLogging(argv[0]);
  // Execute the program logic.
  int ret = combineGramCount();
  // Clean up Google gflags.
  ::gflags::ShutDownCommandLineFlags();
  return ret;
}
