module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @add() -> () {
    %cst1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.splat %cst1 : tensor<2x5xf32>
    %cst2 = arith.constant 2.000000e+00 : f32
    %1 = tensor.splat %cst2 : tensor<2x5xf32>
    %out = tosa.add %0, %1 : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>

    %outb = bufferization.to_buffer %out : tensor<2x5xf32> to memref<2x5xf32>
    %cast = memref.cast %outb : memref<2x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
}

