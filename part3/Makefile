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

	llc -filetype=obj one.ll -o one.o
	g++ one.o main.cpp -o main1

opt:
	$$BINDIR/mlir-opt demo2.mlir \
	-one-shot-bufferize="bufferize-function-boundaries=true" \
	-canonicalize \
	-convert-linalg-to-affine-loops \
	-affine-loop-fusion \
	-affine-loop-tile="cache-size=1024" \
	-affine-loop-normalize \
	-canonicalize -o opt.mlir

	$$BINDIR/mlir-opt opt.mlir \
	-lower-affine \
	-canonicalize \
	-convert-scf-to-cf \
	-convert-func-to-llvm \
	-convert-cf-to-llvm \
	-finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-reconcile-unrealized-casts \
	-canonicalize -o two.mlir

	$$BINDIR/mlir-translate two.mlir \
	--mlir-to-llvmir \
	-o two.ll

	llc -filetype=obj two.ll -o two.o
	g++ two.o main.cpp -o main2

clean:
	rm main1 main2 *.ll *.o one.mlir two.mlir opt.mlir
