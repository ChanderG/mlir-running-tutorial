all: lower compile

# add this to the initial section of the passes
# --llvm-request-c-wrappers
lower:
	$$BINDIR/mlir-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" add.mlir | \
        $$BINDIR/mlir-opt -one-shot-bufferize="bufferize-function-boundaries=true" \
    -canonicalize \
    -convert-linalg-to-loops \
    -convert-scf-to-cf \
    -convert-func-to-llvm \
    -convert-cf-to-llvm \
    -convert-arith-to-llvm \
    -finalize-memref-to-llvm \
    -reconcile-unrealized-casts \
    -canonicalize \
	> add_l.mlir
	$$BINDIR/mlir-translate --mlir-to-llvmir add_l.mlir -o add.ll

compile:
	llc -filetype=obj add.ll -o add.o
	g++ add.o main.cpp -o main

lower_print:
	$$BINDIR/mlir-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" add_p.mlir | \
        $$BINDIR/mlir-opt -one-shot-bufferize="bufferize-function-boundaries=true" \
    -canonicalize \
    -convert-linalg-to-loops \
    -convert-scf-to-cf \
    -convert-func-to-llvm \
    -convert-cf-to-llvm \
    -convert-arith-to-llvm \
    -finalize-memref-to-llvm \
    -reconcile-unrealized-casts \
    -canonicalize > add_p_l.mlir

run_print:
	$$BINDIR/mlir-runner ./add_p_l.mlir -entry-point-result=void -e add -shared-libs=$$BINDIR/../lib/libmlir_runner_utils.so 

clean:
	rm add_p_l.mlir add_l.mlir add.ll add.o main
