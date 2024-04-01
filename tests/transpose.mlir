module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64> loc("tests/transpose.mlir":1:24), %arg1: tensor<*xf64> loc("tests/transpose.mlir":1:27)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("tests/transpose.mlir":2:12)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("tests/transpose.mlir":2:27)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("tests/transpose.mlir":2:25)
    toy.return %2 : tensor<*xf64>loc("tests/transpose.mlir":2:5)
  } loc("tests/transpose.mlir":1:1)
  toy.func @main() {
    %0 = toy.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<2x2xf64> loc("tests/transpose.mlir":6:19)
    %1 = toy.reshape(%0 : tensor<2x2xf64>) to tensor<2x2xf64> loc("tests/transpose.mlir":6:5)
    %2 = toy.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64> loc("tests/transpose.mlir":7:19)
    %3 = toy.reshape(%2 : tensor<4xf64>) to tensor<2x2xf64> loc("tests/transpose.mlir":7:5)
    %4 = toy.generic_call @multiply_transpose(%0, %2) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("tests/transpose.mlir":8:13)
    toy.print %4 loc("tests/transpose.mlir":9:5)
    toy.return loc(unknown)
  } loc("tests/transpose.mlir":5:1)
} loc(unknown)
