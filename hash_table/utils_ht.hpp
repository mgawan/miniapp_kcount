struct Supermer {
  // qualities must be represented, but only as good or bad, so this is done with lowercase for bad, uppercase otherwise
  string seq;
  uint16_t count;

  void pack(const string &unpacked_seq);

  void unpack();

  int get_bytes();
};