#include "iostream"
#include <cstdint>
#include <memory>
#include <random>
#include <cassert>
#include <cstring>
#include <chrono>

using namespace std;

mt19937 rng;

extern "C" {
  void func_l(void*, void*, void*);
  void func_o(void*, void*, void*);
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

void calcReference(Tensor2F*c, Tensor2F *a, Tensor2F* b) {
  auto x = a->sizes[0];
  auto y = a->sizes[1];

  auto y_ = b->sizes[0];
  auto z = b->sizes[1];

  assert(y == y_ && "Matrices dimensions should match up for matmul");

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
      set(c, i, k, acc);
    }
  }

  return;
}
auto timeF =
    [](auto&& func, auto&&... params) {
        // get time before function invocation
        auto start = chrono::high_resolution_clock::now();
        // function invocation using perfect forwarding
        forward<decltype(func)>(func)(forward<decltype(params)>(params)...);
        // get time after function invocation
        auto end = chrono::high_resolution_clock::now();
        return (end - start) / 1ms;
     };

int main() {
  rng.seed(42);

  auto in1 = createTensor2F(256, 512);
  auto in2 = createTensor2F(512, 1024);
  Tensor2F res1, res2;
  // for the reference output
  auto ref = createTensor2F(256, 1024);

  initRandomTensor2F(&in1);
  initRandomTensor2F(&in2);

  cout << "ref: " << timeF(calcReference, &ref, &in1, &in2) << " ms" << endl;
  cout << "lower: " << timeF(func_l, &res1, &in1, &in2) << " ms" << endl;
  cout << "opt: " << timeF(func_o, &res2, &in1, &in2) << " ms" << endl;
}
