# miniapp_kcount
miniapp for GPU kcount portion of metahipmer pipeline
### BUILDING on SPOCK
```bash
module load cmake
module load rocm

mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=hipcc ../
make
```

### RUNNING on Spock
Obtain a GPU node on cluster:

```bash
srun -n1 ./main_app ../test_data/reads_10k.fasta ./ht_out ./pnp_out
```
*ht_out* and *pnp_out* contain the output from hash table insertion and parse and pack kernels respectively. These files can be sorted and compared agains the ground truth files available in the *test_data* folder.
Larger KMER runs can be performed after changing the *KMER_LEN* macro in *main.cpp* file and building again.

