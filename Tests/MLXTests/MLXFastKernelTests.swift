// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

#if canImport(Metal)
    import Metal
#endif

class MLXFastKernelTests: XCTestCase {

    func testCustomKernelBasic() {
        // based on def test_custom_kernel_basic
        MLXRandom.seed(7)
        let a = normal([2, 2])
        let kernel = MLXFast.metalKernel(
            name: "basic",
            inputNames: ["a"],
            outputNames: ["out1"],
            source: """
                    uint elem = thread_position_in_grid.x;
                    out1[elem] = a[elem];
                """)

        let out = kernel(
            [a],
            grid: (4, 1, 1),
            threadGroup: (2, 1, 1),
            outputShapes: [[2, 2]],
            outputDTypes: [.float32])

        XCTAssertTrue(allClose(out[0], a).all().item())
    }

    func testCustomKernelArgs() {
        // based on def test_custom_kernel_args
        MLXRandom.seed(7)
        let a = normal([3, 6])
        let c = normal([2, 2]).asType(.bfloat16)

        let kernel = MLXFast.metalKernel(
            name: "arg_test",
            inputNames: ["a", "b", "c", "d"],
            outputNames: ["out1", "out2"],
            source: """
                    uint elem = thread_position_in_grid.x;
                    T tmp = a[0];
                    if (e) {
                        out1[elem] = a[1] + b[2] + c[3] + d + f;
                    } else {
                        out1[elem] = 1;
                    }
                    out2[elem] = a[1] + b[2] + c[1] - d;
                """)

        let out = kernel(
            [
                a,
                MLXArray([3, 4, 5]),
                c,
                7.3,
            ],
            template: [
                ("e", true),
                ("f", 3),
                ("T", DType.float16),
            ],
            grid: (6, 1, 1),
            threadGroup: (2, 1, 1),
            outputShapes: [[2, 2], [3, 2]],
            outputDTypes: [.float32, .int32])

        XCTAssertTrue(allClose(out[0], full([2, 2], values: 14.0484)).all().item())
        XCTAssertTrue(allClose(out[1], full([3, 2], values: -2)).all().item())
    }

    func testFastSDPA() {
        // https://github.com/ml-explore/mlx-swift/issues/172
        // this will just make sure the MLXFast.scaled_dot_product_attention is
        // callable in the various cases, based on
        // https://github.com/ml-explore/mlx/blob/main/python/tests/test_fast_sdpa.py#L65-L87

        let Dk = 64
        let scale = 1.0 / sqrt(Float(Dk))
        let dTypes = [DType.float32, DType.float16]
        for SEQUENCE_LENGTH in [63, 129, 400] {
            for dtype in dTypes {
                let B = 2
                let H = 24
                let q = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)
                let k = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)
                let v = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)

                let result = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: nil,
                    memoryEfficientThreshold: 2)

                eval(result)
            }
        }
    }

    func testFastSDPAOutput() {
        MLXRandom.seed(0)
        let queries = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 1, 80])
        let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 9, 80])
        let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 9, 80])
        let result = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 0.1118, mask: .none)
        print(result.shape)
        print(result.sum().item(Float.self))
        XCTAssertEqual(result.shape, [1, 32, 1, 80])
        XCTAssertEqual(result.sum().item(Float.self), 1281.9253, accuracy: 0.01)
    }

    func testFastSDPAHead256AgainstCPU() {
        assertFullAttentionAgainstCPU(headDimension: 256, valueDimension: 256)
    }

    func testFastSDPAHead512AgainstCPU() throws {
        #if canImport(Metal)
            guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
                throw XCTSkip("D512 fused attention requires NAX support")
            }
            try XCTSkipUnless(
                MTLCreateSystemDefaultDevice()?.supportsFamily(.apple10) == true,
                "D512 fused attention requires an Apple10 GPU")
            assertFullAttentionAgainstCPU(headDimension: 512, valueDimension: 512)
        #else
            throw XCTSkip("D512 fused attention requires Metal NAX")
        #endif
    }

    func testFastSDPADifferentValueDimensionAgainstCPU() {
        assertFullAttentionAgainstCPU(headDimension: 96, valueDimension: 64)
    }

    private func assertFullAttentionAgainstCPU(
        headDimension: Int, valueDimension: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let scale = 1 / Float(headDimension).squareRoot()
        for dtype in [DType.float16, .bfloat16] {
            for length in [64, 1024] {
                for causal in [false, true] {
                    let (q, k, v, reference) = Device.withDefaultDevice(.cpu) {
                        let q = MLXRandom.normal(
                            [1, 2, length, headDimension], key: MLXRandom.key(51)
                        ).asType(dtype)
                        let k = MLXRandom.normal(
                            [1, 1, length, headDimension], key: MLXRandom.key(52)
                        ).asType(dtype)
                        let v = MLXRandom.normal(
                            [1, 1, length, valueDimension], key: MLXRandom.key(53)
                        ).asType(dtype)
                        var scores =
                            matmul(q.asType(.float32), k.asType(.float32).swappedAxes(-1, -2))
                            * scale
                        if causal {
                            let mask = tril(ones([length, length])).asType(.bool)
                            scores = which(mask, scores, -Float.infinity)
                        }
                        let reference = matmul(softmax(scores, axis: -1), v.asType(.float32))
                        eval(q, k, v, reference)
                        return (q, k, v, reference)
                    }
                    let result = Device.withDefaultDevice(.gpu) {
                        MLXFast.scaledDotProductAttention(
                            queries: q, keys: k, values: v, scale: scale,
                            mask: causal ? .causal : .none, forceFused: true)
                    }.asType(.float32)
                    let difference = result - reference
                    let error = (difference * difference).sum().sqrt().item(Float.self)
                    let norm = (reference * reference).sum().sqrt().item(Float.self)
                    let context =
                        "D=\(headDimension) V=\(valueDimension) L=\(length) \(dtype) causal=\(causal)"
                    XCTAssertEqual(result.shape, reference.shape, context, file: file, line: line)
                    XCTAssertTrue(
                        result.asArray(Float.self).allSatisfy { $0.isFinite }, context,
                        file: file, line: line)
                    XCTAssertLessThan(
                        error / max(norm, 1e-6), 0.02, context, file: file, line: line)
                }
            }
        }
    }

    func testRoPEOutput() {
        // https://github.com/ml-explore/mlx-swift/issues/315
        MLXRandom.seed(0)
        let rope = RoPE(dimensions: 32, traditional: false, base: 10_000, scale: 1)
        let queries = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 1, 80])
        print(queries.shape)
        let result = rope(queries, offset: 8)
        XCTAssertEqual(result.sum().item(Float.self), 1079.7894, accuracy: 0.01)
    }
}
