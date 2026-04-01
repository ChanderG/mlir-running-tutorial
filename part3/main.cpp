#include "iostream"
#include <cstdint>
#include <memory>
#include <random>

using namespace std;

mt19937 rng;

extern "C" {
  void _mlir_ciface_myfunc(void*, void*, void*);
}

template<typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

typedef struct MemRefDescriptor<float, 2> Tensor2F;

Tensor2F createTensor2F(int x, int y) {
  Tensor2F t;

  // how much size we need
  auto num_bytes = x*y*sizeof(float);
  // how much we need to take extra to ensure our 64 byte boundary
  auto nb_full = num_bytes + 64;

  char *curr;
  curr = new char[nb_full];
  t.allocated = (float *) curr;
  t.aligned = (float*) (curr + (64 - (uintptr_t)curr%64));

  t.offset = 0;
  t.sizes[0] = x;
  t.sizes[1] = y;
  t.strides[0] = y;
  t.strides[1] = 1;

  return t;
}

void initRandomTensor2F(Tensor2F *t) {
  // Not the most efficient, but ok for now
  float* base = (float*)((uintptr_t*)t->aligned + t->offset);
  auto x = t->sizes[0];
  auto y = t->sizes[1];
  auto xs = t->strides[0];
  auto ys = t->strides[1];

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      base[i*xs + j*ys] = 1.0*rng()/(rng.max() - rng.min());
    }
  }
}

int main() {
  rng.seed(42);

  auto t = createTensor2F(2, 5);
  initRandomTensor2F(&t);

  cout << "Tensor is: ";
  for (int i = 0; i < 10; i++) {
    cout << t.aligned[i] << " ";
  }
  cout << endl;
}
