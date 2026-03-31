#include "iostream"
#include <cstdint>
#include <memory>
using namespace std;

extern "C" {
  void _mlir_ciface_add(void*);
}

template<typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

int main() {
  MemRefDescriptor<float, 2> res;
  _mlir_ciface_add(&res);

  cout << "Tensor is: " << res.sizes[0] << "x" << res.sizes[1] << endl;
  cout << "Strides are: " << res.strides[0] << "x" << res.strides[1] << endl;
  cout << "Offset is: "  << res.offset << endl;

  cout << "Some sample values are: " << endl;
  for (int i = 0; i < 10; i++) {
    cout << res.aligned[i] << " ";
  }
  cout << endl;

  cout << "Allocated: " << (uintptr_t)res.allocated;
  cout << " Aligned: " << (uintptr_t)res.aligned << endl;
  cout << "Diff is: " << res.aligned - res.allocated << endl;
}
