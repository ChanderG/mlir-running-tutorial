lower:
	$$BINDIR/mlir-opt demo2.mlir \
	-one-shot-bufferize="bufferize-function-boundaries=true" \
	-canonicalize \
	-convert-linalg-to-loops \
	-convert-scf-to-cf \
	-convert-func-to-llvm \
	-convert-cf-to-llvm \
	-finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-reconcile-unrealized-casts \
	-canonicalize -o one.mlir

	$$BINDIR/mlir-translate one.mlir \
	--mlir-to-llvmir \
	-o one.ll

compile:
	llc -filetype=obj one.ll -o one.o
	g++ one.o main.cpp -o main
