#include "iostream"
#include <cstdint>
#include <memory>
using namespace std;

extern "C" {
  void _mlir_ciface_add(void*, void*, void*);
}

// alignas() only works for static allocations

template<typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

int main() {
  MemRefDescriptor<float, 2> in1, in2, res;

  // we will go for a 2x5 tensor

  // how much size we need
  auto num_bytes = 2*5*sizeof(float);
  // how much we need to take extra to ensure our 64 byte boundary
  auto nb_full = num_bytes + 64;

  char *curr;
  curr = new char[nb_full];
  in1.allocated = (float *) curr;
  in1.aligned = (float*) (curr + (64 - (uintptr_t)curr%64));

  curr = new char[nb_full];
  in2.allocated = (float *) curr;
  in2.aligned = (float*) (curr + (64 - (uintptr_t)curr%64));

  in1.offset = in2.offset = 0;
  in1.sizes[0] = in2.sizes[0] = 2;
  in1.sizes[1] = in2.sizes[1] = 5;
  in1.strides[0] = in2.strides[0] = 5;
  in1.strides[1] = in2.strides[1] = 1;

  // fill in the values
  fill(in1.aligned, in1.aligned+10, 2.0);
  fill(in2.aligned, in2.aligned+10, 3.0);

  _mlir_ciface_add(&res, &in1, &in2);

  cout << "Tensor is: " << res.sizes[0] << "x" << res.sizes[1] << endl;
  cout << "Strides are: " << res.strides[0] << "x" << res.strides[1] << endl;
  cout << "Offset is: "  << res.offset << endl;

  for (int i = 0; i < 10; i++) {
    cout << res.aligned[i] << " ";
  }
  cout << endl;
}
