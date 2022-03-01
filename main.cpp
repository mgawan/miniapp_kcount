#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "./parse_and_pack.hpp"
#include "utils_ht.hpp"

#define QUAL_OFFSET 0
#define MINIMIZER_LEN 100
#define KMER_LEN 33
#define KCOUNT_SEQ_BLOCK_SIZE 3000000
#define MAX_K 33

const int N_LONGS = (MAX_K + 31) / 32;

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
    std::vector<std::string> reads;
    std::string   read_in;
    std::string seq_block_in;
    std::string in_file = argv[1];

    uint32_t dropped = 0;

     uint32_t kmer_max = 1000000; 
    // size_t gpu_avail_mem = 8500000000; // about 8 GB.
    // size_t gpu_bytes_req = 0;
    // ht_gpu_driver.init(0, 1, KMER_LEN, kmer_max, gpu_avail_mem, gpu_bytes_req);


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

    std::cout << "total read size in:"<<read_size_in<<std::endl;
    std::cout << "block size:"<< seq_block_in.size()<<std::endl;
    std::cout << "total reads:" << reads.size() << std::endl;
    int minimizer_len = KMER_LEN * 2 / 3 + 1;
    std::cout << "Minimizer len:"<< minimizer_len << std::endl;
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
    
    KmerDHT<MAX_K> kmer_dht(kmer_max);
  
    for (int i = 0; i < num_targets; i++) {
        Supermer supermer;
        auto target = pnp_gpu_driver.supermers[i].target;
        auto offset = pnp_gpu_driver.supermers[i].offset;
        auto len = pnp_gpu_driver.supermers[i].len;

         std::cout <<" Target:"<<target<< " offset:"<< offset << " len:" << len<< std::endl;

       // std::string supermer_seq;
        
        int packed_len = len / 2;
        if (offset % 2 || len % 2) packed_len++;
        supermer.seq = pnp_gpu_driver.packed_seqs.substr(offset / 2, packed_len);
        if (offset % 2) supermer.seq[0] &= 15;
        if ((offset + len) % 2) supermer.seq[supermer.seq.length() - 1] &= 240;
        
        supermer.count = (uint16_t)1;
        kmer_dht.add_supermer(supermer, target);
        kmer_dht.ht_gpu_driver.insert_supermer(supermer.seq, supermer.count);

        std::cout <<" seq:"<< supermer.seq << " len:" << len<< std::endl;
  }
  int num_dropped = 0, num_unique = 0, num_purged = 0;
  kmer_dht.ht_gpu_driver.insert_supermer_block(); // this should launch the kernel
  kmer_dht.ht_gpu_driver.done_all_inserts(num_dropped, num_unique, num_purged);



}