module {
  func.func @add(%arg0: tensor<2x5xf32>, %arg1: tensor<2x5xf32>) -> tensor<2x5xf32> attributes {llvm.emit_c_interface} {
    %out = tosa.add %arg0, %arg1 : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    return %out : tensor<2x5xf32>
  }
}