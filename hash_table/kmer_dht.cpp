/*
 HipMer v 2.0, Copyright (c) 2020, The Regents of the University of California,
 through Lawrence Berkeley National Laboratory (subject to receipt of any required
 approvals from the U.S. Dept. of Energy).  All rights reserved."

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 (1) Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 (2) Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 (3) Neither the name of the University of California, Lawrence Berkeley National
 Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to
 endorse or promote products derived from this software without specific prior
 written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 DAMAGE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades
 to the features, functionality or performance of the source code ("Enhancements") to
 anyone; however, if you choose to make your Enhancements available either publicly,
 or directly to Lawrence Berkeley National Laboratory, without imposing a separate
 written license agreement for such Enhancements, then you hereby grant the following
 license: a  non-exclusive, royalty-free perpetual license to install, use, modify,
 prepare derivative works, incorporate into other computer software, distribute, and
 sublicense such enhancements or derivative works thereof, in binary and source code
 form.
*/

#include <stdarg.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include "kmer_dht.hpp"

using namespace std;

size_t estimate_hashtable_memory(size_t num_elements, size_t element_size) {
  // get the hashtable load factor
  std::unordered_map<char, char> tmp;
  double max_load_factor = tmp.max_load_factor();

  // apply the load factor
  size_t expanded_num_elements = num_elements / max_load_factor + 1;

  // get the next power of two
  --expanded_num_elements;
  expanded_num_elements |= expanded_num_elements >> 1;
  expanded_num_elements |= expanded_num_elements >> 2;
  expanded_num_elements |= expanded_num_elements >> 4;
  expanded_num_elements |= expanded_num_elements >> 8;
  expanded_num_elements |= expanded_num_elements >> 16;
  expanded_num_elements |= expanded_num_elements >> 32;
  ++expanded_num_elements;

  size_t num_buckets = expanded_num_elements * max_load_factor;

  return expanded_num_elements * element_size + num_buckets * 8;
}

template <int MAX_K>
void KmerArray<MAX_K>::set(const uint64_t *kmer) {
  memcpy(longs, kmer, N_LONGS * sizeof(uint64_t));
}

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
      // default: DIE("Invalid value encountered when packing '", unpacked_seq[i], "' ", (int)unpacked_seq[i]);
    };
    seq[i / 2] |= (!(i % 2) ? (packed_val << 4) : packed_val);
    // if (seq[i / 2] == '_') DIE("packed byte is same as sentinel _");
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

template <int MAX_K>
KmerDHT<MAX_K>::KmerDHT(uint64_t my_num_kmers)
    : local_kmers({})
    , my_num_kmers(my_num_kmers){

  size_t gpu_avail_mem = 8500000000; // about 8 GB.
  size_t gpu_bytes_req = 0;
  int temp_kmer_len = MAX_K;
  ht_gpu_driver.init(0, 1, temp_kmer_len, my_num_kmers, gpu_avail_mem, gpu_bytes_req);
  // minimizer len depends on k
  minimizer_len =  temp_kmer_len * 2 / 3 + 1;
  if (minimizer_len < 15) minimizer_len = 15;
  if (minimizer_len > 27) minimizer_len = 27;
  
  auto node0_cores = 1;//upcxx::local_team().rank_n();
  // check if we have enough memory to run - conservative because we don't want to run out of memory
  double adjustment_factor = 0.2;
  auto my_adjusted_num_kmers = my_num_kmers * adjustment_factor;
  double required_space = estimate_hashtable_memory(my_adjusted_num_kmers, sizeof(Kmer<MAX_K>) + sizeof(KmerCounts)) * node0_cores;
  std::cout << " estimated space for hash table:"<< required_space << std::endl;
  // auto max_reqd_space = required_space;
 // auto free_mem = get_free_mem();
 // auto lowest_free_mem = free_mem;//upcxx::reduce_all(free_mem, upcxx::op_fast_min).wait();
 // auto highest_free_mem = free_mem;//upcxx::reduce_all(free_mem, upcxx::op_fast_max).wait();
  // SLOG_VERBOSE("With adjustment factor of ", adjustment_factor, " require ", get_size_str(max_reqd_space), " per node (",
  //              my_adjusted_num_kmers, " kmers per rank), and there is ", get_size_str(lowest_free_mem), " to ",
  //              get_size_str(highest_free_mem), " available on the nodes\n");
 // if (lowest_free_mem * 0.80 < max_reqd_space) SWARN("Insufficient memory available: this could crash with OOM (lowest=", get_size_str(lowest_free_mem), " vs reqd=", get_size_str(max_reqd_space), ")");

  //kmer_store.set_size("kmers", max_kmer_store_bytes, max_rpcs_in_flight, useHHSS);

  //barrier();
  // in this case we have to roughly estimate the hash table size because the space is reserved now
  // err on the side of excess because the whole point of doing this is speed and we don't want a
  // hash table resize
  // Unfortunately, this estimate depends on the depth of the sample - high depth means more wasted memory,
  // but low depth means potentially resizing the hash table, which is very expensive
  double kmers_space_reserved = my_adjusted_num_kmers * (sizeof(Kmer<MAX_K>) + sizeof(KmerCounts));
  my_adjusted_num_kmers *= 4;

}

template <int MAX_K>
KmerDHT<MAX_K>::~KmerDHT() {
  // local_kmers->clear();
  // KmerMap<MAX_K>().swap(*local_kmers);
  // clear_stores();
}

template <int MAX_K>
void KmerDHT<MAX_K>::init_ctg_kmers(int64_t max_elems) {
  using_ctg_kmers = true;
}

template <int MAX_K>
int KmerDHT<MAX_K>::get_minimizer_len() {
  return minimizer_len;
}

#define KMER_DHT_K(KMER_LEN) template class KmerDHT<KMER_LEN>

KMER_DHT_K(32);
#if MAX_BUILD_KMER >= 64
KMER_DHT_K(64);
#endif
#if MAX_BUILD_KMER >= 96
KMER_DHT_K(96);
#endif
#if MAX_BUILD_KMER >= 128
KMER_DHT_K(128);
#endif
#if MAX_BUILD_KMER >= 160
KMER_DHT_K(160);
#endif

#undef KMER_DHT_K
