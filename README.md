# miniapp_kcount
miniapp for GPU kcount portion of metahipmer pipeline
### BUILDING on CORI
```bash
module load cmake
module load cuda

mkdir build
cd build
cmake ../
make
```

### RUNNING on CORI
Obtain a GPU node on Cori GPU cluster:

```bash
srun -n1 ./main_app ../test_data/reads_10k.fasta ./ht_out ./pnp_out
```
*ht_out* and *pnp_out* contain the output from hash table insertion and parse and pack kernels respectively. These files can be sorted and compared agains the ground truth files available in the *test_data* folder.
Larger KMER runs can be performed after changing the *KMER_LEN* macro in *main.cpp* file and building again.


