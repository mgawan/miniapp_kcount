#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "./parse_and_pack.hpp"

#define QUAL_OFFSET 10
#define MINIMIZER_LEN 100
#define KMER_LEN 33
#define KCOUNT_SEQ_BLOCK_SIZE 3000000
#define MAX_K 77

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

    double driver_init_time = 0;
    kcount_gpu::ParseAndPackGPUDriver pnp_gpu_driver(0, 1, QUAL_OFFSET, KMER_LEN, N_LONGS, MINIMIZER_LEN, driver_init_time);
    uint32_t num_valid_kmers = 0;
    pnp_gpu_driver.process_seq_block(seq_block_in, num_valid_kmers);

    if (pnp_gpu_driver.kernel_is_done()){
        pnp_gpu_driver.pack_seq_block(seq_block_in);
    }

    int num_targets = (int)pnp_gpu_driver.supermers.size();
    
    for (int i = 0; i < num_targets; i++) {
        auto target = pnp_gpu_driver.supermers[i].target;
        auto offset = pnp_gpu_driver.supermers[i].offset;
        auto len = pnp_gpu_driver.supermers[i].len;
        std::string supermer_seq;
        
        int packed_len = len / 2;
        if (offset % 2 || len % 2) packed_len++;
        supermer_seq = pnp_gpu_driver.packed_seqs.substr(offset / 2, packed_len);
        if (offset % 2) supermer_seq[0] &= 15;
        if ((offset + len) % 2) supermer_seq[supermer_seq.length() - 1] &= 240;
        std::cout <<" Target:"<< i << " seq:"<< supermer_seq<< std::endl;
  }


}