#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "parse_and_pack.hpp"
#include "kmer_dht.hpp"
#include "gpu_hash_table.hpp"
#include <stdlib.h> 
#include <time.h> 

//** DMAX_BUILD_KMER needs to be adjusted in CMakedefs to a larger value for larger KMER_LENS **
#define QUAL_OFFSET 0
#define KMER_LEN 32
#define KCOUNT_SEQ_BLOCK_SIZE 3000000
#define MAX_K KMER_LEN

const int N_LONGS = (MAX_BUILD_KMER + 31) / 32;

void process_seq(std::string &seq, std::string &seq_block, uint32_t &dropped) {
    if(seq_block.size() > KCOUNT_SEQ_BLOCK_SIZE){
        dropped++;
        std::cout << "WARNING: seq block size exceeding the limit, numbers of reads dropped:"<<dropped<< std::endl;
        return;
    }
    seq_block += seq;
    seq_block += "_";  
}

int main (int argc, char* argv[]){
    srand (time(NULL));
    std::vector<std::string> reads;
    std::string   read_in;
    std::string seq_block_in;
    std::string in_file = argv[1];
    std::string hash_t_out = argv[2];
    std::string pnp_out = argv[3];

    uint32_t dropped = 0;

    std::ifstream read_file(in_file);
    uint32_t read_size_in = 0;

    if(read_file.is_open()){
        while(getline(read_file, read_in)){  
            if(read_in[0] == '>')
                continue;
            else{
                reads.push_back(read_in);
                read_size_in += read_in.size();
            }
        }
    }

    for(auto seq : reads){
        process_seq(seq, seq_block_in, dropped);
    }

    uint32_t total_kmers_est = (300 - MAX_K + 1 )* reads.size();
    uint32_t kmer_max = total_kmers_est;
    std::cout << "estimted kmers:" << total_kmers_est << std::endl;
    
    std::cout << "total read size in:"<<read_size_in<<std::endl;
    std::cout << "block size:"<< seq_block_in.size()<<std::endl;
    std::cout << "total reads:" << reads.size() << std::endl;
    int minimizer_len = MAX_BUILD_KMER * 2 / 3 + 1;
    // std::cout << "Minimizer len:"<< minimizer_len << std::endl;
    double driver_init_time = 0;
    kcount_gpu::ParseAndPackGPUDriver pnp_gpu_driver(0, 1, QUAL_OFFSET, KMER_LEN, N_LONGS, minimizer_len, driver_init_time);
    uint32_t num_valid_kmers = 0;
    auto status = pnp_gpu_driver.process_seq_block(seq_block_in, num_valid_kmers);

    if(status == 2){
        std::cout<< "ERR: kernel launch in process_seq_block failed!" << std::endl;
        std::cout << "length too large" << std::endl;
    }else if ( status == 3){
        std::cout<< "ERR: kernel launch in process_seq_block failed!" << std::endl;
        std::cout << "length zero" << std::endl; 
    }else if (status == 4){
        std::cout<< "ERR: kernel launch in process_seq_block failed!" << std::endl;
        std::cout << "length less than kmer length" << std::endl;
    }else{
        std::cout << "INFO: Kernel launch success!"<<std::endl;
    }

    if (pnp_gpu_driver.kernel_synch()){
        pnp_gpu_driver.pack_seq_block(seq_block_in);
    }else{
        std::cout<< "ERR: pack kernel was not launched" << std::endl;
    }

    int num_targets = (int)pnp_gpu_driver.supermers.size();
    
    KmerDHT<MAX_K> kmer_dht(kmer_max); // initializes the hash table driver, reserves space on GPU and CPU for packed sequeces.
    uint32_t targets_ctgs = num_targets * 1.0;
    std::cout << "INFO: total targets:" << num_targets << " Targets to be used as contigs:" << targets_ctgs << std::endl;
    
    std::ofstream pnp_file(pnp_out);

    for (int i = 0; i < num_targets ; i++) {
        Supermer supermer;
        auto target = pnp_gpu_driver.supermers[i].target;
        auto offset = pnp_gpu_driver.supermers[i].offset;
        auto len = pnp_gpu_driver.supermers[i].len;

 // the below code/conversions are copied as is from metahipmer
        int packed_len = len / 2;
        if (offset % 2 || len % 2) packed_len++;
        supermer.seq = pnp_gpu_driver.packed_seqs.substr(offset / 2, packed_len);
        if (offset % 2) supermer.seq[0] &= 15;
        if ((offset + len) % 2) supermer.seq[supermer.seq.length() - 1] &= 240;
        
        pnp_file <<" Target:"<< target << " seq:"<< supermer.seq<< std::endl;
        supermer.count = (uint16_t)1;
        kmer_dht.ht_gpu_driver.insert_supermer(supermer.seq, supermer.count);

  }
  pnp_file.flush();
  pnp_file.close();

  kmer_dht.ht_gpu_driver.insert_supermer_block(); // launch insertion kernel for read kmers
  auto stats_kmer = kmer_dht.ht_gpu_driver.get_stats();

  std::cout << "\nGPU STATS FROM READ KMER INSERTS:" << std::endl;
  std::cout << "DROPPED:" << stats_kmer.dropped << "\nATTEMPTED:" << stats_kmer.attempted << "\nNEW INSERTS:" << stats_kmer.new_inserts << "\nKEY EMPTY OVERLAPPED:" << stats_kmer.key_empty_overlaps << std::endl;
    


// ***** CTG KMERS PASS *******
  kmer_dht.ht_gpu_driver.init_ctg_kmers(kmer_max, 8500000000);
  for (int i = 0; i < targets_ctgs; i++) {
        Supermer supermer;
        auto target = pnp_gpu_driver.supermers[i].target;
        auto offset = pnp_gpu_driver.supermers[i].offset;
        auto len = pnp_gpu_driver.supermers[i].len;

    // the below code/conversions are copied as is from metahipmer
        int packed_len = len / 2;
        if (offset % 2 || len % 2) packed_len++;
        supermer.seq = pnp_gpu_driver.packed_seqs.substr(offset / 2, packed_len);
        if (offset % 2) supermer.seq[0] &= 15;
        if ((offset + len) % 2) supermer.seq[supermer.seq.length() - 1] &= 240;
        
        supermer.count = (uint16_t) 2;//(rand() % 3);
        kmer_dht.ht_gpu_driver.insert_supermer(supermer.seq, supermer.count);
  }

    int num_dropped = 0, num_unique = 0, num_purged = 0;
    kmer_dht.ht_gpu_driver.insert_supermer_block(); // launch insertion kernel from ctg kmers
    stats_kmer = kmer_dht.ht_gpu_driver.get_stats();

    std::cout << "\nGPU STATS FROM CTG KMER INSERTS:" << std::endl;
    std::cout << "DROPPED:" << stats_kmer.dropped << "\nATTEMPTED:" << stats_kmer.attempted << "\nNEW INSERTS:" << stats_kmer.new_inserts << "\nKEY EMPTY OVERLAPPED:" << stats_kmer.key_empty_overlaps << std::endl;
    

    kmer_dht.ht_gpu_driver.done_ctg_kmer_inserts(num_dropped, num_unique, num_purged); // because we have the contig pass as well

    kmer_dht.ht_gpu_driver.done_all_inserts(num_dropped, num_unique, num_purged);
    kmer_dht.ht_gpu_driver.print_keys_vals(hash_t_out);

    std::cout << "\nSTATS AFTER PURGE:" << std::endl;
    std::cout << "DROPPED:" << num_dropped << "\nUNIQUE:" << num_unique << "\nPURGED:" << num_purged << std::endl;

}
