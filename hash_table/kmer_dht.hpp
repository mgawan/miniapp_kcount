//empty

#pragma once

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

#include <map>
#include <iterator>
#include <string>
// #include <upcxx/upcxx.hpp>

// #include "utils.hpp"
#include "kmer.hpp"
#include "gpu_hash_table.hpp"
// #include "upcxx_utils/flat_aggr_store.hpp"
// #include "upcxx_utils/three_tier_aggr_store.hpp"


using namespace std;
// global variables to avoid passing dist objs to rpcs
int _dmin_thres = 2.0;

// struct FragElem;


// template <int MAX_K>
// struct KmerAndExt {
//   Kmer<MAX_K> kmer;
//   kmer_count_t count;
//   char left, right;
//   UPCXX_SERIALIZED_FIELDS(kmer, count, left, right);
// };


size_t estimate_hashtable_memory(size_t num_elements, size_t element_size);
using count_t = uint32_t;



template <int MAX_K>
class KmerDHT {
 private:
  KmerMap<MAX_K> local_kmers;

  //upcxx_utils::ThreeTierAggrStore<Supermer> kmer_store;
  int64_t max_kmer_store_bytes;
  int64_t my_num_kmers;
  int max_rpcs_in_flight;
  //std::chrono::time_point<std::chrono::high_resolution_clock> start_t;

  int minimizer_len = 15;

 public:

  HashTableGPUDriver<MAX_K> ht_gpu_driver;
  bool using_ctg_kmers = false;
  
  KmerDHT(uint64_t my_num_kmers);

  void clear_stores();

  ~KmerDHT();

  void init_ctg_kmers(int64_t max_elems);

  int get_minimizer_len();

  uint64_t get_num_kmers(bool all = false);

  int64_t get_local_num_kmers(void);

  //upcxx::intrank_t get_kmer_target_rank(const Kmer<MAX_K> &kmer, const Kmer<MAX_K> *kmer_rc = nullptr) const;

  KmerCounts *get_local_kmer_counts(Kmer<MAX_K> &kmer);

  // bool kmer_exists(Kmer<MAX_K> kmer);

  // void add_supermer(Supermer &supermer, int target_rank);

  // void flush_updates();

  // void finish_updates();

  // one line per kmer, format:
  // KMERCHARS LR N
  // where L is left extension and R is right extension, one char, either X, F or A, C, G, T
  // where N is the count of the kmer frequency
  //void dump_kmers();

  typename KmerMap<MAX_K>::iterator local_kmers_begin();

  typename KmerMap<MAX_K>::iterator local_kmers_end();

  int32_t get_time_offset_us();
};

