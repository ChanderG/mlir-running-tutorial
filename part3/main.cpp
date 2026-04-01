#include "iostream"
#include <cstdint>
#include <memory>
#include <random>
#include <cassert>
#include <cstring>

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

float get(Tensor2F *t, int i, int j) {
  // Not the most efficient, but ok for now
  float* base = (float*)((uintptr_t*)t->aligned + t->offset);
  return base[i*t->strides[0] + j*t->strides[1]];
}

void set(Tensor2F *t, int i, int j, float val) {
  // Not the most efficient, but ok for now
  float* base = (float*)((uintptr_t*)t->aligned + t->offset);
  base[i*t->strides[0] + j*t->strides[1]] = val;
}

Tensor2F calcReference(Tensor2F *a, Tensor2F* b) {
  auto x = a->sizes[0];
  auto y = a->sizes[1];

  auto y_ = b->sizes[0];
  auto z = b->sizes[1];

  assert(y == y_ && "Matrices dimensions should match up for matmul");

  auto c = createTensor2F(x, z);

  // naive implementation - can take a lot of time
  for (int i = 0; i < x; i++) {
    for (int k = 0; k < z; k++) {
      // calc result for c[i, k]
      float acc = 0.0;
      for (int j = 0; j < y; j++) {
        acc += get(a, i, j)*get(b, j, k);
      }

      // let's do the fused relu here
      if (acc < 0) {
        acc = 0;
      }
      set(&c, i, k, acc);
    }
  }

  return c;
}

int main() {
  rng.seed(42);

  auto in1 = createTensor2F(256, 512);
  auto in2 = createTensor2F(512, 1024);
  Tensor2F res;

  initRandomTensor2F(&in1);
  initRandomTensor2F(&in2);

  // calc naive reference value
  auto ref = calcReference(&in1, &in2);

  _mlir_ciface_myfunc(&res, &in1, &in2);

  cout << "Tensor is: " << res.sizes[0] << "x" << res.sizes[1] << endl;
  cout << "Strides are: " << res.strides[0] << "x" << res.strides[1] << endl;

  auto cmp = memcmp(res.aligned, ref.aligned, ref.sizes[0]*ref.sizes[1]*sizeof(float));
  if (cmp == 0) {
    cout << "MLIR version matches reference!" << endl;
  } else {
    cout << "result != reference !!!" << endl;
  }
}
