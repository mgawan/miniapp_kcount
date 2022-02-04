
#include "utils_ht.hpp"

void Supermer::pack(const string &unpacked_seq) {
  // each position in the sequence is an upper or lower case nucleotide, not including Ns
  seq = string(unpacked_seq.length() / 2 + unpacked_seq.length() % 2, 0);
  for (int i = 0; i < unpacked_seq.length(); i++) {
    char packed_val = 0;
    switch (unpacked_seq[i]) {
      case 'a': packed_val = 1; break;
      case 'c': packed_val = 2; break;
      case 'g': packed_val = 3; break;
      case 't': packed_val = 4; break;
      case 'A': packed_val = 5; break;
      case 'C': packed_val = 6; break;
      case 'G': packed_val = 7; break;
      case 'T': packed_val = 8; break;
      case 'N': packed_val = 9; break;
      default: DIE("Invalid value encountered when packing '", unpacked_seq[i], "' ", (int)unpacked_seq[i]);
    };
    seq[i / 2] |= (!(i % 2) ? (packed_val << 4) : packed_val);
    if (seq[i / 2] == '_') DIE("packed byte is same as sentinel _");
  }
}

void Supermer::unpack() {
  static const char to_base[] = {0, 'a', 'c', 'g', 't', 'A', 'C', 'G', 'T', 'N'};
  string unpacked_seq;
  for (int i = 0; i < seq.length(); i++) {
    unpacked_seq += to_base[(seq[i] & 240) >> 4];
    int right_ext = seq[i] & 15;
    if (right_ext) unpacked_seq += to_base[right_ext];
  }
  seq = unpacked_seq;
}

int Supermer::get_bytes() { return seq.length() + sizeof(kmer_count_t); }
