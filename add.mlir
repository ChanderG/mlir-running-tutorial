module {
  func.func @add() -> tensor<2x5xf32> attributes {llvm.emit_c_interface} {
    %cst1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.splat %cst1 : tensor<2x5xf32>
    %cst2 = arith.constant 2.000000e+00 : f32
    %1 = tensor.splat %cst2 : tensor<2x5xf32>
    %out = tosa.add %0, %1 : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    return %out : tensor<2x5xf32>
  }
}

