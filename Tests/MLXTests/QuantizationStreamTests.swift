// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

// The producer stops using the array before the consumer takes it.
private final class QuantizedMatmulHandoff: @unchecked Sendable {
    private let lock = NSLock()
    private var output: MLXArray?
    let ready = DispatchSemaphore(value: 0)

    func store(_ output: MLXArray) {
        lock.withLock { self.output = output }
        ready.signal()
    }

    func take() -> MLXArray? {
        lock.withLock {
            defer { output = nil }
            return output
        }
    }
}

@Suite("Quantized matmul streams")
struct QuantizationStreamTests {
    @Test func mixedDtypeMatmulCanBeEvaluatedOnAnotherThread() async {
        await #expect(processExitsWith: .success) {
            for dtype in [DType.float32, .bfloat16] {
                let handoff = QuantizedMatmulHandoff()
                Thread {
                    let stream = StreamOrDevice.stream(MLX.Stream(.cpu))
                    let input = MLXArray.ones([1, 128], dtype: dtype, stream: stream)
                    let weight = MLXArray.zeros([1, 8], dtype: .uint32, stream: stream)
                    let scales = MLXArray.ones([1, 1], dtype: .float16, stream: stream)
                    let biases = MLXArray.ones([1, 1], dtype: .float16, stream: stream)
                    eval(input, weight, scales, biases)
                    handoff.store(
                        quantizedMM(
                            input, weight, scales: scales, biases: biases,
                            groupSize: 128, bits: 2, stream: stream))
                }.start()

                guard handoff.ready.wait(timeout: .now() + 10) == .success,
                    let output = handoff.take()
                else {
                    Issue.record("The quantized matmul producer did not finish")
                    return
                }
                #expect(output.dtype == .float32)
                #expect(output.item(Float.self) == 128)
            }
        }
    }
}
