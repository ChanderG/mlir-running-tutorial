lower:	
	$$BINDIR/mlir-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" add.mlir | \
	$$BINDIR/mlir-opt -one-shot-bufferize="bufferize-function-boundaries=true" -convert-linalg-to-loops -convert-scf-to-cf \
	-convert-func-to-llvm -convert-cf-to-llvm \
        -convert-arith-to-llvm -finalize-memref-to-llvm \
	-reconcile-unrealized-casts -canonicalize > add_l.mlir
	$$BINDIR/mlir-translate --mlir-to-llvmir add_l.mlir -o add.ll

compile:
	llc -filetype=obj add.ll -o add.o
	g++ add.o main.cpp -o main

clean:
	rm add_l.mlir add.ll add.o main
