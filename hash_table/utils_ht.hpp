#include "kmer.hpp"

size_t estimate_hashtable_memory(size_t num_elements, size_t element_size);
using count_t = uint32_t;

template <int MAX_K>
struct KmerArray {
  static const int N_LONGS = (MAX_K + 31) / 32;
  uint64_t longs[N_LONGS];

  void set(const uint64_t *x);
};
struct CountsArray {
  count_t kmer_count;
  ext_count_t ext_counts[8];
};

struct Supermer {
  // qualities must be represented, but only as good or bad, so this is done with lowercase for bad, uppercase otherwise
  string seq;
  uint16_t count;

  void pack(const string &unpacked_seq);
  void unpack();
  int get_bytes();
};

struct FramElem;

struct KmerCounts {
  FragElem uutig_frag;
  // how many times this kmer has occurred: don't need to count beyond 65536
  kmer_count_t count;
  // the final extensions chosen - A,C,G,T, or F,X
  char left, right;
};

template <int MAX_K>
using KmerMap = HASH_TABLE<Kmer<MAX_K>, KmerCounts>;


template <int MAX_K>
class KmerDHT {
 private:
  KmerMap<MAX_K> local_kmers;
  HashTableGPUDriver<MAX_K>> ht_gpu_driver;

  //upcxx_utils::ThreeTierAggrStore<Supermer> kmer_store;
  int64_t max_kmer_store_bytes;
  int64_t my_num_kmers;
  int max_rpcs_in_flight;
  //std::chrono::time_point<std::chrono::high_resolution_clock> start_t;

  int minimizer_len = 15;

 public:
  bool using_ctg_kmers = false;
  
  KmerDHT(uint64_t my_num_kmers, int max_rpcs_in_flight, bool useHHSS);

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

template <int MAX_K>
struct KmerCountsMap {
  // Arrays for keys and values. They are separate because the keys get initialized with max number and the vals with zero
  KmerArray<MAX_K> *keys = nullptr;
  CountsArray *vals = nullptr;
  int64_t capacity = 0;
  int num = 0;

  void init(int64_t ht_capacity);
  void clear();
};

struct SupermerBuff {
  char *seqs;
  count_t *counts;
};

template <int MAX_K>
struct KmerCountsMap {
  // Arrays for keys and values. They are separate because the keys get initialized with max number and the vals with zero
  KmerArray<MAX_K> *keys = nullptr;
  CountsArray *vals = nullptr;
  int64_t capacity = 0;
  int num = 0;

  void init(int64_t ht_capacity);
  void clear();
};

template <int MAX_K>
struct KmerExtsMap {
  KmerArray<MAX_K> *keys = nullptr;
  CountExts *vals = nullptr;
  int64_t capacity = 0;

  void init(int64_t ht_capacity);
  void clear();
};

struct InsertStats {
  unsigned int dropped = 0;
  unsigned int attempted = 0;
  unsigned int new_inserts = 0;
  unsigned int key_empty_overlaps = 0;
};

template <int MAX_K>
class HashTableGPUDriver {
  static const int N_LONGS = (MAX_K + 31) / 32;
  //struct HashTableDriverState;
  // stores CUDA specific variables
 // HashTableDriverState *dstate = nullptr;

  // int upcxx_rank_me;
  // int upcxx_rank_n;
  int kmer_len;
  int buff_len = 0;
  std::vector<KmerArray<MAX_K>> output_keys;
  std::vector<CountExts> output_vals;
  size_t output_index = 0;

  KmerCountsMap<MAX_K> read_kmers_dev;
  KmerCountsMap<MAX_K> ctg_kmers_dev;

  // for buffering elements in the host memory
  SupermerBuff elem_buff_host = {0};
  // for transferring host memory buffer to device
  SupermerBuff unpacked_elem_buff_dev = {0};
  SupermerBuff packed_elem_buff_dev = {0};

  InsertStats read_kmers_stats;
  InsertStats ctg_kmers_stats;
  InsertStats *gpu_insert_stats;
  int num_gpu_calls = 0;

  void insert_supermer_block();
  void purge_invalid(int &num_purged, int &num_entries);

 public:
  PASS_TYPE pass_type;

  HashTableGPUDriver();
  ~HashTableGPUDriver();

  void init(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int max_elems, size_t gpu_avail_mem, size_t &gpu_bytes_reqd);

  void init_ctg_kmers(int max_elems, size_t gpu_avail_mem);

  void insert_supermer(const std::string &supermer_seq, count_t supermer_count);

  void flush_inserts();

  void done_ctg_kmer_inserts(int &attempted_inserts, int &dropped_inserts, int &new_inserts);

  void done_all_inserts(int &num_dropped, int &num_unique, int &num_purged);

  std::pair<KmerArray<MAX_K> *, CountExts *> get_next_entry();

  void get_elapsed_time(double &insert_time, double &kernel_time);

  int64_t get_capacity();

  InsertStats &get_stats();

  int get_num_gpu_calls();
};
