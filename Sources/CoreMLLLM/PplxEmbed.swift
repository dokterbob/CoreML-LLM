import CoreML
import Foundation
import Tokenizers

/// Runtime for Perplexity's pplx-embed (a bidirectional Qwen3 encoder → masked
/// mean pool → tanh int8 quantize). Exposes the official pplx-embed contract:
///
///   plain   : `[String] -> [[Int8]]`        (1024-d int8 per text)
///   context : `[[String]] -> [[[Int8]]]`    (per-document late chunking;
///             per-chunk 1024-d int8)
///
/// Each call also exposes the `binary` (+1/-1 Float) and `ubinary` (packed
/// UInt8[dim/8]) variants.
///
/// Output-format design decision. The underlying mlpackage emits native int8
/// already (the `int8` output IS the deliverable; it's readable in Swift via
/// the dtype-agnostic NSNumber subscript even though the Python CoreML bridge
/// can't read int8 on macOS 26). We derive the other two formats directly from
/// the int8 vector:
///
///   binary[i]  = int8[i] >= 0 ? +1 : -1
///   ubinary    = packbits(int8[i] >= 0)
///
/// This is bit-exact with the reference `st_quantize` everywhere except the
/// measure-zero x≈0 case: the reference branches on the raw pre-tanh value
/// `x >= 0`, whereas we branch on the rounded int8. Since `round(tanh(x)*127)`
/// is 0 only in a tiny neighbourhood of x=0 and is otherwise sign-faithful,
/// the int8-derived sign agrees with the raw sign except when |x| is so small
/// it rounds to int8 0 — there we map 0 to the `>= 0` (positive) branch to
/// match the reference's tie direction. For strictly bit-exact binary/ubinary
/// against a `pooled_fp16`-output model, build with
/// `--output-mode pooled_fp16` and apply all three quantizers in Swift; we ship
/// the int8-derived path because it needs only one model and one forward pass.
///
/// I/O contract of the underlying mlpackages (from build_pplx_embed_bundle.py):
///   plain:
///     input_ids       (1, L)   int32
///     attention_mask  (1, L)   fp16   (1.0 valid, 0.0 pad)
///     → embedding     (1, 1024) int8
///   context:
///     input_ids       (1, L)   int32
///     attention_mask  (1, L)   fp16
///     pool_matrix     (32, L)  fp16   (row k = 1/n_k over chunk k's span)
///     → embedding     (32, 1024) int8 (only first n_chunks rows are valid)
public final class PplxEmbed {

    /// The three published pplx-embed output formats.
    public enum Format: String, Sendable {
        case int8
        case binary
        case ubinary
    }

    /// Per-bundle config parsed from model_config.json.
    public struct BucketConfig: Sendable {
        public let maxSeqLen: Int
        public let embedDim: Int
        public let variant: String   // "plain" | "context"
        public let url: URL
    }

    public static let embedDim = 1024
    public static let nMaxChunks = 32

    private let tokenizer: Tokenizer
    private let sepTokenId: Int
    private let computeUnits: MLComputeUnits

    /// Available buckets, sorted ascending by maxSeqLen.
    private let buckets: [BucketConfig]
    private let variant: String

    /// Lazily compiled+loaded models, keyed by bucket maxSeqLen.
    private var loaded: [Int: MLModel] = [:]
    private let lock = NSLock()

    private init(tokenizer: Tokenizer, sepTokenId: Int, buckets: [BucketConfig],
                 variant: String, computeUnits: MLComputeUnits) {
        self.tokenizer = tokenizer
        self.sepTokenId = sepTokenId
        self.buckets = buckets
        self.variant = variant
        self.computeUnits = computeUnits
    }

    // MARK: - Loading

    /// Load a pplx-embed bundle.
    ///
    /// `bundleDir` may be either:
    ///   * a directory of bucket subdirectories (e.g. `output/pplx-embed/`
    ///     containing `L512-int8/`, `L1024-int8/`, …) — all int8 buckets are
    ///     discovered and used for token-length-based bucket selection, or
    ///   * a single bucket directory directly containing `encoder.mlpackage`
    ///     (e.g. `output/pplx-embed-context/L512-int8/`).
    ///
    /// Models are compiled + loaded lazily on first use per bucket; the
    /// tokenizer is loaded eagerly from the first bucket's `hf_model/`.
    public static func load(
        bundleDir: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> PplxEmbed {
        let fm = FileManager.default
        var buckets: [BucketConfig] = []

        // A single bucket dir directly contains encoder.mlpackage / .mlmodelc.
        let isSingle = fm.fileExists(atPath: bundleDir.appendingPathComponent("encoder.mlpackage").path)
            || fm.fileExists(atPath: bundleDir.appendingPathComponent("encoder.mlmodelc").path)

        if isSingle {
            if let c = parseBucket(at: bundleDir) { buckets.append(c) }
        } else {
            let entries = (try? fm.contentsOfDirectory(at: bundleDir,
                includingPropertiesForKeys: nil)) ?? []
            for e in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                guard (try? e.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
                else { continue }
                if let c = parseBucket(at: e) { buckets.append(c) }
            }
        }

        guard !buckets.isEmpty else {
            throw CoreMLLLMError.modelNotFound(
                "no pplx-embed bucket with encoder.mlpackage/.mlmodelc under \(bundleDir.path)")
        }

        let variant = buckets.first!.variant
        // Keep only buckets that match the dominant variant, sorted ascending.
        let sorted = buckets.filter { $0.variant == variant }
            .sorted { $0.maxSeqLen < $1.maxSeqLen }

        let hfDir = sorted.first!.url.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: hfDir)
        let sepId = sepTokenId(fromHFDir: hfDir) ?? 151643

        return PplxEmbed(tokenizer: tokenizer, sepTokenId: sepId, buckets: sorted,
                         variant: variant, computeUnits: computeUnits)
    }

    /// Parse a single bucket directory's model_config.json. Only accepts
    /// int8-output buckets (the deliverable format).
    private static func parseBucket(at dir: URL) -> BucketConfig? {
        let fm = FileManager.default
        let hasModel = fm.fileExists(atPath: dir.appendingPathComponent("encoder.mlpackage").path)
            || fm.fileExists(atPath: dir.appendingPathComponent("encoder.mlmodelc").path)
        guard hasModel,
              fm.fileExists(atPath: dir.appendingPathComponent("hf_model").path)
        else { return nil }

        let cfgURL = dir.appendingPathComponent("model_config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        let outputMode = j["output_mode"] as? String ?? "int8"
        guard outputMode == "int8" else { return nil }   // skip pooled_fp16 / quantized variants

        let maxSeqLen = (j["bucket"] as? Int) ?? (j["max_seq_len"] as? Int) ?? 512
        let embedDim = (j["hidden_size"] as? Int) ?? PplxEmbed.embedDim
        let variant = (j["variant"] as? String)
            ?? (dir.path.contains("context") ? "context" : "plain")

        return BucketConfig(maxSeqLen: maxSeqLen, embedDim: embedDim,
                            variant: variant, url: dir)
    }

    private static func sepTokenId(fromHFDir hfDir: URL) -> Int? {
        // <|endoftext|> id from added_tokens.json (pplx tokenizer: 151643).
        let url = hfDir.appendingPathComponent("added_tokens.json")
        guard let data = try? Data(contentsOf: url),
              let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let id = j["<|endoftext|>"] as? Int
        else { return nil }
        return id
    }

    private func model(forBucket L: Int) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = loaded[L] { return m }
        guard let cfg = buckets.first(where: { $0.maxSeqLen == L }) else {
            throw CoreMLLLMError.modelNotFound("no loaded bucket for L=\(L)")
        }
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        let compiled = cfg.url.appendingPathComponent("encoder.mlmodelc")
        let pkg = cfg.url.appendingPathComponent("encoder.mlpackage")
        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            modelURL = compiled
        } else {
            modelURL = try compileSync(pkg)
        }
        let m = try MLModel(contentsOf: modelURL, configuration: mlConfig)
        loaded[L] = m
        return m
    }

    /// Synchronous compile wrapper (MLModel.compileModel is async on newer SDKs
    /// but the legacy throwing overload is sync). Use the sync overload to keep
    /// the model accessor non-async.
    private func compileSync(_ pkg: URL) throws -> URL {
        try MLModel.compileModel(at: pkg)
    }

    /// Pick the smallest bucket whose maxSeqLen >= n; if none, the largest.
    private func bucket(forTokens n: Int) -> BucketConfig {
        for b in buckets where b.maxSeqLen >= n { return b }
        return buckets.last!
    }

    // MARK: - Plain API

    /// Encode texts into 1024-d int8 embeddings (one row per text).
    public func embed(_ texts: [String]) throws -> [[Int8]] {
        try texts.map { try embedOne($0) }
    }

    /// Encode texts and return the requested format.
    /// - int8:    `[[Int8]]` (1024-d)
    /// - binary:  `[[Float]]` (1024-d, +1/-1)
    /// - ubinary: `[[UInt8]]` (128 packed bytes)
    public func embedInt8(_ texts: [String]) throws -> [[Int8]] {
        try embed(texts)
    }

    public func embedBinary(_ texts: [String]) throws -> [[Float]] {
        try embed(texts).map { PplxEmbed.binary(fromInt8: $0) }
    }

    public func embedUBinary(_ texts: [String]) throws -> [[UInt8]] {
        try embed(texts).map { PplxEmbed.ubinary(fromInt8: $0) }
    }

    private func embedOne(_ text: String) throws -> [Int8] {
        var ids = tokenizer.encode(text: text)
        let bucket = bucket(forTokens: ids.count)
        let L = bucket.maxSeqLen
        if ids.count > L { ids = Array(ids.prefix(L)) }
        let n = ids.count

        let inputIds = try makeInputIds(ids, L: L)
        let attn = try makeAttentionMask(n: n, L: L)

        let model = try model(forBucket: L)
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIds,
            "attention_mask": attn,
        ]))
        guard let arr = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        // (1, 1024) int8 → first 1024 values.
        let d = min(PplxEmbed.embedDim, arr.count)
        var vec = [Int8](repeating: 0, count: d)
        for i in 0..<d { vec[i] = Int8(arr[i].int8Value) }
        return vec
    }

    // MARK: - Context API (late chunking)

    /// Late-chunking context embed. For each document (a list of chunk strings),
    /// returns per-chunk 1024-d int8 embeddings: `[[Int8]]` with one row per
    /// chunk, in order.
    public func embedContext(_ documents: [[String]]) throws -> [[[Int8]]] {
        try documents.map { try embedContextOne($0) }
    }

    public func embedContextBinary(_ documents: [[String]]) throws -> [[[Float]]] {
        try embedContext(documents).map { doc in doc.map { PplxEmbed.binary(fromInt8: $0) } }
    }

    public func embedContextUBinary(_ documents: [[String]]) throws -> [[[UInt8]]] {
        try embedContext(documents).map { doc in doc.map { PplxEmbed.ubinary(fromInt8: $0) } }
    }

    private func embedContextOne(_ chunks: [String]) throws -> [[Int8]] {
        precondition(variant == "context",
                     "embedContext requires a context bundle (variant=context)")
        guard !chunks.isEmpty else { return [] }

        // Join chunks with the sep token, then tokenize the whole window once.
        // The tokenizer adds the literal <|endoftext|> between chunks; we locate
        // its ids among the valid tokens to recover chunk spans.
        let sep = "<|endoftext|>"
        let joined = chunks.joined(separator: sep)
        var ids = tokenizer.encode(text: joined)

        let bucket = bucket(forTokens: ids.count)
        let L = bucket.maxSeqLen
        if ids.count > L { ids = Array(ids.prefix(L)) }
        let n = ids.count

        // Recover chunk spans: [start, sep) (SEP excluded), next start = sep+1,
        // final chunk runs to n.
        var spans: [(Int, Int)] = []
        var start = 0
        for i in 0..<n where ids[i] == sepTokenId {
            spans.append((start, i))
            start = i + 1
        }
        spans.append((start, n))
        // Cap at the model's max chunk count.
        if spans.count > PplxEmbed.nMaxChunks {
            spans = Array(spans.prefix(PplxEmbed.nMaxChunks))
        }
        let nChunks = spans.count

        let inputIds = try makeInputIds(ids, L: L)
        let attn = try makeAttentionMask(n: n, L: L)
        let pool = try makePoolMatrix(spans: spans, L: L)

        let model = try model(forBucket: L)
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIds,
            "attention_mask": attn,
            "pool_matrix": pool,
        ]))
        guard let arr = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        // (32, 1024) int8 — read only the first nChunks rows (rest are all-zero).
        let D = PplxEmbed.embedDim
        var result: [[Int8]] = []
        result.reserveCapacity(nChunks)
        for c in 0..<nChunks {
            var row = [Int8](repeating: 0, count: D)
            let base = c * D
            for i in 0..<D { row[i] = Int8(arr[base + i].int8Value) }
            result.append(row)
        }
        return result
    }

    // MARK: - Input builders

    private func makeInputIds(_ ids: [Int], L: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
        let p = arr.dataPointer.bindMemory(to: Int32.self, capacity: L)
        for i in 0..<L { p[i] = i < ids.count ? Int32(ids[i]) : 0 }
        return arr
    }

    private func makeAttentionMask(n: Int, L: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .float16)
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: L)
        let one: UInt16 = 0x3C00  // 1.0 in fp16
        for i in 0..<L { p[i] = i < n ? one : 0 }
        return arr
    }

    /// (32, L) fp16 pool matrix; row k = 1/n_k over chunk k's [start,end) span,
    /// unused rows all-zero.
    private func makePoolMatrix(spans: [(Int, Int)], L: Int) throws -> MLMultiArray {
        let rows = PplxEmbed.nMaxChunks
        let arr = try MLMultiArray(shape: [NSNumber(value: rows), NSNumber(value: L)],
                                   dataType: .float16)
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: rows * L)
        for i in 0..<(rows * L) { p[i] = 0 }
        for (k, span) in spans.enumerated() where k < rows {
            let (s, e) = span
            let count = e - s
            guard count > 0 else { continue }
            let w = float16Bits(Float(1.0) / Float(count))
            let base = k * L
            for col in s..<e { p[base + col] = w }
        }
        return arr
    }

    // MARK: - Format derivation

    /// binary[i] = int8[i] >= 0 ? +1 : -1   (matches reference x>=0 branch).
    public static func binary(fromInt8 v: [Int8]) -> [Float] {
        v.map { $0 >= 0 ? Float(1) : Float(-1) }
    }

    /// ubinary = packbits(int8[i] >= 0), MSB-first per byte (numpy packbits).
    public static func ubinary(fromInt8 v: [Int8]) -> [UInt8] {
        let nBytes = (v.count + 7) / 8
        var out = [UInt8](repeating: 0, count: nBytes)
        for i in 0..<v.count where v[i] >= 0 {
            out[i / 8] |= UInt8(1 << (7 - (i % 8)))
        }
        return out
    }

    /// Float → IEEE-754 binary16 bit pattern (native Float16 round).
    private func float16Bits(_ x: Float) -> UInt16 {
        Float16(x).bitPattern
    }
}
