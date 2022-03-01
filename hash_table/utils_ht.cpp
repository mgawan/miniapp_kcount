
#include "utils_ht.hpp"

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

template <int MAX_K>
KmerDHT<MAX_K>::KmerDHT(uint64_t my_num_kmers)
    : local_kmers({})
    , my_num_kmers(my_num_kmers){

  size_t gpu_avail_mem = 8500000000; // about 8 GB.
  size_t gpu_bytes_req = 0;
  ht_gpu_driver.init(0, 1, KMER_LEN, my_num_kmers, gpu_avail_mem, gpu_bytes_reqd);
  // minimizer len depends on k
  minimizer_len =  KMER_LEN * 2 / 3 + 1;
  if (minimizer_len < 15) minimizer_len = 15;
  if (minimizer_len > 27) minimizer_len = 27;
 // SLOG_VERBOSE("Using a minimizer length of ", minimizer_len, "\n");
  // main purpose of the timer here is to track memory usage
 // BarrierTimer timer(__FILEFUNC__);
  auto node0_cores = 1;//upcxx::local_team().rank_n();
  // check if we have enough memory to run - conservative because we don't want to run out of memory
  double adjustment_factor = 0.2;
  auto my_adjusted_num_kmers = my_num_kmers * adjustment_factor;
  double required_space = estimate_hashtable_memory(my_adjusted_num_kmers, sizeof(Kmer<MAX_K>) + sizeof(KmerCounts)) * node0_cores;
  auto max_reqd_space = required_space;
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
  // SLOG_VERBOSE("Reserving at least ", get_size_str(node0_cores * kmers_space_reserved), " for kmer hash tables with ",
  //              node0_cores * my_adjusted_num_kmers, " entries on node 0\n");
  //double init_free_mem = get_free_mem();
  if (my_adjusted_num_kmers <= 0) DIE("no kmers to reserve space for");
  // kmer_store.set_update_func([&ht_inserter = this->ht_inserter](Supermer supermer) {
  //   num_inserts++;
  //   ht_inserter->insert_supermer(supermer.seq, supermer.count);
  // });
  my_adjusted_num_kmers *= 4;
  //ht_inserter->init(my_adjusted_num_kmers);
  //barrier();
}

template <int MAX_K>
void KmerDHT<MAX_K>::clear_stores() {
  kmer_store.clear();
}

template <int MAX_K>
KmerDHT<MAX_K>::~KmerDHT() {
  local_kmers->clear();
  KmerMap<MAX_K>().swap(*local_kmers);
  clear_stores();
}

template <int MAX_K>
void KmerDHT<MAX_K>::init_ctg_kmers(int64_t max_elems) {
  using_ctg_kmers = true;
  //ht_inserter->init_ctg_kmers(max_elems);
}

template <int MAX_K>
int KmerDHT<MAX_K>::get_minimizer_len() {
  return minimizer_len;
}

template <int MAX_K>
uint64_t KmerDHT<MAX_K>::get_num_kmers(bool all) {
  if (!all)
    return (uint64_t)local_kmers->size();
  else
    return (uint64_t)local_kmers->size();
}

template <int MAX_K>
int64_t KmerDHT<MAX_K>::get_local_num_kmers(void) {
  return local_kmers->size();
}

// template <int MAX_K>
// upcxx::intrank_t KmerDHT<MAX_K>::get_kmer_target_rank(const Kmer<MAX_K> &kmer, const Kmer<MAX_K> *kmer_rc) const {
//   assert(&kmer != kmer_rc && "Can be a palindrome, cannot be the same Kmer instance");
//   return kmer.minimizer_hash_fast(minimizer_len, kmer_rc) % rank_n();
// }

template <int MAX_K>
KmerCounts *KmerDHT<MAX_K>::get_local_kmer_counts(Kmer<MAX_K> &kmer) {
  const auto it = local_kmers->find(kmer);
  if (it == local_kmers->end()) return nullptr;
  return &it->second;
}

// template <int MAX_K>
// bool KmerDHT<MAX_K>::kmer_exists(Kmer<MAX_K> kmer_fw) {
//   const Kmer<MAX_K> kmer_rc = kmer_fw.revcomp();
//   const Kmer<MAX_K> *kmer = (kmer_rc < kmer_fw) ? &kmer_rc : &kmer_fw;

//   return rpc(
//              get_kmer_target_rank(kmer_fw, &kmer_rc),
//              [](Kmer<MAX_K> kmer, dist_object<KmerMap<MAX_K>> &local_kmers) -> bool {
//                const auto it = local_kmers->find(kmer);
//                if (it == local_kmers->end()) return false;
//                return true;
//              },
//              *kmer, local_kmers)
//       .wait();
// }

// template <int MAX_K>
// void KmerDHT<MAX_K>::add_supermer(Supermer &supermer, int target_rank) {
//   kmer_store.update(target_rank, supermer);
// }

// template <int MAX_K>
// void KmerDHT<MAX_K>::flush_updates() {
//   BarrierTimer timer(__FILEFUNC__);
//   kmer_store.flush_updates();
//   barrier();
//   ht_inserter->flush_inserts();
// }

// template <int MAX_K>
// void KmerDHT<MAX_K>::finish_updates() {
//   ht_inserter->insert_into_local_hashtable(local_kmers);
//   double insert_time, kernel_time;
//   ht_inserter->get_elapsed_time(insert_time, kernel_time);
//   stage_timers.kernel_kmer_analysis->inc_elapsed(kernel_time);
// }

// one line per kmer, format:
// KMERCHARS LR N
// where L is left extension and R is right extension, one char, either X, F or A, C, G, T
// where N is the count of the kmer frequency
// template <int MAX_K>
// void KmerDHT<MAX_K>::dump_kmers() {
//   //BarrierTimer timer(__FILEFUNC__);
//   int k = Kmer<MAX_K>::get_k();
//   //string dump_fname = "kmers-" + to_string(k) + ".txt.gz";
//   //get_rank_path(dump_fname, rank_me());
//   //zstr::ofstream dump_file(dump_fname);
//   ostringstream out_buf;
//   ProgressBar progbar(local_kmers->size(), "Dumping kmers to " + dump_fname);
//   int64_t i = 0;
//   for (auto &elem : *local_kmers) {
//     out_buf << elem.first << " " << elem.second.count << " " << elem.second.left << " " << elem.second.right;
//     out_buf << endl;
//     i++;
//     if (!(i % 1000)) {
//       dump_file << out_buf.str();
//       out_buf = ostringstream();
//     }
//     progbar.update();
//   }
//   if (!out_buf.str().empty()) dump_file << out_buf.str();
//   dump_file.close();
//   progbar.done();
//   SLOG_VERBOSE("Dumped ", this->get_num_kmers(), " kmers\n");
// }

template <int MAX_K>
typename KmerMap<MAX_K>::iterator KmerDHT<MAX_K>::local_kmers_begin() {
  return local_kmers->begin();
}

template <int MAX_K>
typename KmerMap<MAX_K>::iterator KmerDHT<MAX_K>::local_kmers_end() {
  return local_kmers->end();
}

template <int MAX_K>
int32_t KmerDHT<MAX_K>::get_time_offset_us() {
  std::chrono::duration<double> t_elapsed = CLOCK_NOW() - start_t;
  return std::chrono::duration_cast<std::chrono::microseconds>(t_elapsed).count();
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

template <int MAX_K>
void KmerCountsMap<MAX_K>::init(int64_t ht_capacity) {
  capacity = ht_capacity;
  cudaErrchk(cudaMalloc(&keys, capacity * sizeof(KmerArray<MAX_K>)));
  cudaErrchk(cudaMemset((void *)keys, KEY_EMPTY_BYTE, capacity * sizeof(KmerArray<MAX_K>)));
  cudaErrchk(cudaMalloc(&vals, capacity * sizeof(CountsArray)));
  cudaErrchk(cudaMemset(vals, 0, capacity * sizeof(CountsArray)));
}

template <int MAX_K>
void KmerCountsMap<MAX_K>::clear() {
  cudaFree((void *)keys);
  cudaFree(vals);
}


template <int MAX_K>
HashTableGPUDriver<MAX_K>::HashTableGPUDriver() {}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::init(int upcxx_rank_me, int upcxx_rank_n, int kmer_len, int max_elems, size_t gpu_avail_mem, size_t &gpu_bytes_reqd) {
  //QuickTimer init_timer;
  //init_timer.start();
  this->upcxx_rank_me = upcxx_rank_me;
  this->upcxx_rank_n = upcxx_rank_n;
  this->kmer_len = kmer_len;
  pass_type = READ_KMERS_PASS;
  gpu_utils::set_gpu_device(upcxx_rank_me);
  // now check that we have sufficient memory for the required capacity
  size_t elem_buff_size = KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * (1 + sizeof(count_t)) * 1.5;
  size_t elem_size = sizeof(KmerArray<MAX_K>) + sizeof(CountsArray);
  gpu_bytes_reqd = (max_elems * elem_size) / 0.8 + elem_buff_size;
  // save 1/5 of avail gpu memory for possible ctg kmers and compact hash table
  // set capacity to max avail remaining from gpu memory - more slots means lower load
  auto max_slots = 0.8 * (gpu_avail_mem - elem_buff_size) / elem_size;
  // find the first prime number lower than this value
  primes::Prime prime;
  prime.set(min((size_t)max_slots, (size_t)(max_elems * 3)), false);
  auto ht_capacity = prime.get();
  read_kmers_dev.init(ht_capacity);
  // for transferring packed elements from host to gpu
  elem_buff_host.seqs = new char[KCOUNT_GPU_HASHTABLE_BLOCK_SIZE];
  // these are not used for kmers from reads
  elem_buff_host.counts = nullptr;
  // buffer on the device
  cudaErrchk(cudaMalloc(&packed_elem_buff_dev.seqs, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE));
  cudaErrchk(cudaMalloc(&unpacked_elem_buff_dev.seqs, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * 2));
  packed_elem_buff_dev.counts = nullptr;
  unpacked_elem_buff_dev.counts = nullptr;

  cudaErrchk(cudaMalloc(&gpu_insert_stats, sizeof(InsertStats)));
  cudaErrchk(cudaMemset(gpu_insert_stats, 0, sizeof(InsertStats)));

  //dstate = new HashTableDriverState();
  //init_timer.stop();
  //init_time = init_timer.get_elapsed();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::init_ctg_kmers(int max_elems, size_t gpu_avail_mem) {
  pass_type = CTG_KMERS_PASS;
  size_t elem_buff_size = KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * (1 + sizeof(count_t)) * 1.5;
  size_t elem_size = sizeof(KmerArray<MAX_K>) + sizeof(CountsArray);
  size_t max_slots = 0.8 * (gpu_avail_mem - elem_buff_size) / elem_size;
  primes::Prime prime;
  prime.set(min(max_slots, (size_t)(max_elems * 3)), false);
  auto ht_capacity = prime.get();
  ctg_kmers_dev.init(ht_capacity);
  elem_buff_host.counts = new count_t[KCOUNT_GPU_HASHTABLE_BLOCK_SIZE];
  cudaErrchk(cudaMalloc(&packed_elem_buff_dev.counts, KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * sizeof(count_t)));
  cudaErrchk(cudaMalloc(&unpacked_elem_buff_dev.counts, 2 * KCOUNT_GPU_HASHTABLE_BLOCK_SIZE * sizeof(count_t)));
  cudaErrchk(cudaMemset(gpu_insert_stats, 0, sizeof(InsertStats)));
}

template <int MAX_K>
HashTableGPUDriver<MAX_K>::~HashTableGPUDriver() {
  if (dstate) delete dstate;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_supermer_block() {
  //dstate->insert_timer.start();
  bool is_ctg_kmers = false;//(pass_type == CTG_KMERS_PASS);
  cudaErrchk(cudaMemcpy(packed_elem_buff_dev.seqs, elem_buff_host.seqs, buff_len, cudaMemcpyHostToDevice));
  cudaErrchk(cudaMemset(unpacked_elem_buff_dev.seqs, 0, buff_len * 2));
  if (is_ctg_kmers)
    cudaErrchk(cudaMemcpy(packed_elem_buff_dev.counts, elem_buff_host.counts, buff_len * sizeof(count_t), cudaMemcpyHostToDevice));

  int gridsize, threadblocksize;
  //dstate->kernel_timer.start();
  get_kernel_config(buff_len, gpu_unpack_supermer_block, gridsize, threadblocksize);
  gpu_unpack_supermer_block<<<gridsize, threadblocksize>>>(unpacked_elem_buff_dev, packed_elem_buff_dev, buff_len);
  get_kernel_config(buff_len * 2, gpu_insert_supermer_block<MAX_K>, gridsize, threadblocksize);
  gpu_insert_supermer_block<<<gridsize, threadblocksize>>>(is_ctg_kmers ? ctg_kmers_dev : read_kmers_dev, unpacked_elem_buff_dev,
                                                           buff_len * 2, kmer_len, is_ctg_kmers, gpu_insert_stats);
  // the kernel time is not going to be accurate, because we are not waiting for the kernel to complete
  // need to uncomment the line below, which will decrease performance by preventing the overlap of GPU and CPU execution
  // cudaDeviceSynchronize();
  //dstate->kernel_timer.stop();
 // num_gpu_calls++;
  //dstate->insert_timer.stop();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::insert_supermer(const string &supermer_seq, count_t supermer_count) {
  if (buff_len + supermer_seq.length() + 1 >= KCOUNT_GPU_HASHTABLE_BLOCK_SIZE) {
    insert_supermer_block();
    buff_len = 0;
  }
  memcpy(&(elem_buff_host.seqs[buff_len]), supermer_seq.c_str(), supermer_seq.length());
  // if (pass_type == CTG_KMERS_PASS) {
  //   for (int i = 0; i < (int)supermer_seq.length(); i++) elem_buff_host.counts[buff_len + i] = supermer_count;
  // }
  buff_len += supermer_seq.length();
  elem_buff_host.seqs[buff_len] = '_';
  if (pass_type == CTG_KMERS_PASS) elem_buff_host.counts[buff_len] = 0;
  buff_len++;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::purge_invalid(int &num_purged, int &num_entries) {
  num_purged = num_entries = 0;
  unsigned int *counts_gpu;
  int NUM_COUNTS = 2;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  //GPUTimer t;
  int gridsize, threadblocksize;
  get_kernel_config(read_kmers_dev.capacity, gpu_purge_invalid<MAX_K>, gridsize, threadblocksize);
  //t.start();
  // now purge all invalid kmers (do it on the gpu)
  gpu_purge_invalid<<<gridsize, threadblocksize>>>(read_kmers_dev, counts_gpu);
  //t.stop();
  //dstate->kernel_timer.inc(t.get_elapsed());

  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  num_purged = counts_host[0];
  num_entries = counts_host[1];
  auto expected_num_entries = read_kmers_stats.new_inserts - num_purged;
  // if (num_entries != (int)expected_num_entries)
  //   cout << KLRED << "[" << upcxx_rank_me << "] WARNING mismatch " << num_entries << " != " << expected_num_entries << " diff "
  //        << (num_entries - (int)expected_num_entries) << " new inserts " << read_kmers_stats.new_inserts << " num purged "
  //        << num_purged << KNORM << endl;
  read_kmers_dev.num = num_entries;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::flush_inserts() {
  if (buff_len) {
    insert_supermer_block();
    buff_len = 0;
  }
  cudaErrchk(cudaMemcpy(pass_type == READ_KMERS_PASS ? &read_kmers_stats : &ctg_kmers_stats, gpu_insert_stats, sizeof(InsertStats),
                        cudaMemcpyDeviceToHost));
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_all_inserts(int &num_dropped, int &num_unique, int &num_purged) {
  int num_entries = 0;
  purge_invalid(num_purged, num_entries);
  read_kmers_dev.num = num_entries;
  if (elem_buff_host.seqs) delete[] elem_buff_host.seqs;
  if (elem_buff_host.counts) delete[] elem_buff_host.counts;
  cudaFree(packed_elem_buff_dev.seqs);
  cudaFree(unpacked_elem_buff_dev.seqs);
  if (packed_elem_buff_dev.counts) cudaFree(packed_elem_buff_dev.counts);
  if (unpacked_elem_buff_dev.counts) cudaFree(unpacked_elem_buff_dev.counts);
  cudaFree(gpu_insert_stats);
  // overallocate to reduce collisions
  num_entries *= 1.3;
  // now compact the hash table entries
  unsigned int *counts_gpu;
  int NUM_COUNTS = 2;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  KmerExtsMap<MAX_K> compact_read_kmers_dev;
  compact_read_kmers_dev.init(num_entries);
  //GPUTimer t;
  int gridsize, threadblocksize;
  get_kernel_config(read_kmers_dev.capacity, gpu_compact_ht<MAX_K>, gridsize, threadblocksize);
  //t.start();
  gpu_compact_ht<<<gridsize, threadblocksize>>>(read_kmers_dev, compact_read_kmers_dev, counts_gpu);
  //t.stop();
  //dstate->kernel_timer.inc(t.get_elapsed());
  read_kmers_dev.clear();
  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaFree(counts_gpu);
  num_dropped = counts_host[0];
  num_unique = counts_host[1];
  if (num_unique != read_kmers_dev.num)
    cerr << KLRED << "[" << upcxx_rank_me << "] <gpu_hash_table.cpp:" << __LINE__ << "> WARNING: " << KNORM
         << "mismatch in expected entries " << num_unique << " != " << read_kmers_dev.num << "\n";
  // now copy the gpu hash table values across to the host
  // We only do this once, which requires enough memory on the host to store the full GPU hash table, but since the GPU memory
  // is generally a lot less than the host memory, it should be fine.
  output_keys.resize(num_entries);
  output_vals.resize(num_entries);
  output_index = 0;
  cudaErrchk(cudaMemcpy(output_keys.data(), (void *)compact_read_kmers_dev.keys,
                        compact_read_kmers_dev.capacity * sizeof(KmerArray<MAX_K>), cudaMemcpyDeviceToHost));
  cudaErrchk(cudaMemcpy(output_vals.data(), compact_read_kmers_dev.vals, compact_read_kmers_dev.capacity * sizeof(CountExts),
                        cudaMemcpyDeviceToHost));
  compact_read_kmers_dev.clear();
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::done_ctg_kmer_inserts(int &attempted_inserts, int &dropped_inserts, int &new_inserts) {
  unsigned int *counts_gpu;
  int NUM_COUNTS = 3;
  cudaErrchk(cudaMalloc(&counts_gpu, NUM_COUNTS * sizeof(unsigned int)));
  cudaErrchk(cudaMemset(counts_gpu, 0, NUM_COUNTS * sizeof(unsigned int)));
  GPUTimer t;
  int gridsize, threadblocksize;
  get_kernel_config(ctg_kmers_dev.capacity, gpu_merge_ctg_kmers<MAX_K>, gridsize, threadblocksize);
  t.start();
  gpu_merge_ctg_kmers<<<gridsize, threadblocksize>>>(read_kmers_dev, ctg_kmers_dev, counts_gpu);
  t.stop();
  dstate->kernel_timer.inc(t.get_elapsed());
  ctg_kmers_dev.clear();
  unsigned int counts_host[NUM_COUNTS];
  cudaErrchk(cudaMemcpy(&counts_host, counts_gpu, NUM_COUNTS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaFree(counts_gpu);
  attempted_inserts = counts_host[0];
  dropped_inserts = counts_host[1];
  new_inserts = counts_host[2];
  read_kmers_dev.num += new_inserts;
  read_kmers_stats.new_inserts += new_inserts;
}

template <int MAX_K>
void HashTableGPUDriver<MAX_K>::get_elapsed_time(double &insert_time, double &kernel_time) {
  insert_time = dstate->insert_timer.get_elapsed();
  kernel_time = dstate->kernel_timer.get_elapsed();
}

template <int MAX_K>
pair<KmerArray<MAX_K> *, CountExts *> HashTableGPUDriver<MAX_K>::get_next_entry() {
  if (output_keys.empty() || output_index == output_keys.size()) return {nullptr, nullptr};
  output_index++;
  return {&(output_keys[output_index - 1]), &(output_vals[output_index - 1])};
}

template <int MAX_K>
int64_t HashTableGPUDriver<MAX_K>::get_capacity() {
  if (pass_type == READ_KMERS_PASS)
    return read_kmers_dev.capacity;
  else
    return ctg_kmers_dev.capacity;
}

template <int MAX_K>
InsertStats &HashTableGPUDriver<MAX_K>::get_stats() {
  if (pass_type == READ_KMERS_PASS)
    return read_kmers_stats;
  else
    return ctg_kmers_stats;
}

template <int MAX_K>
int HashTableGPUDriver<MAX_K>::get_num_gpu_calls() {
  return num_gpu_calls;
}

template class kcount_gpu::HashTableGPUDriver<32>;
#if MAX_BUILD_KMER >= 64
template class kcount_gpu::HashTableGPUDriver<64>;
#endif
#if MAX_BUILD_KMER >= 96
template class kcount_gpu::HashTableGPUDriver<96>;
#endif
#if MAX_BUILD_KMER >= 128
template class kcount_gpu::HashTableGPUDriver<128>;
#endif
#if MAX_BUILD_KMER >= 160
template class kcount_gpu::HashTableGPUDriver<160>;
#endif
