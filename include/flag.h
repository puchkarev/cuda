#ifndef FLAG_H
#define FLAG_H

#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

char* ParseFlag(int argc, char** argv, const char* flag_name) {
  for (int i = 0; i < argc; ++i) {
    const bool is_flag = argv[i][0] == '-' && argv[i][1] == '-';
    if (!is_flag) continue;
    const int offset = 2;

    for (int ci = 0; ; ++ci) {
      if (argv[i][ci + offset] == '\0') break;
      if (flag_name[ci] == '\0' && argv[i][ci + offset] == '=') {
        return &argv[i][ci+offset+1];
      }
      if (flag_name[ci] != argv[i][ci+offset]) break;
    }
  }
  return nullptr;
}

int64_t ParseIntFlagOrDefault(int argc, char** argv, std::string flag_name, int64_t default_val) {
  char* val_ptr = ParseFlag(argc, argv, flag_name.c_str());
  if (val_ptr == nullptr) {
    return default_val;
  }
  return atoll(val_ptr);
}

#define GetIntFlagOrDefault(flag_name, default_val) ParseIntFlagOrDefault(argc, argv, flag_name, default_val)

#endif // FLAG_H
