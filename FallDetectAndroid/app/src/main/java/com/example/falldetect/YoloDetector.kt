package com.example.falldetect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

const val TAG = "FallDetect"

/** Format des 4 premières valeurs de chaque détection en sortie du modèle. */
enum class BoxFormat { XYXY, XYWH }

/** Boîte normalisée (0..1) dans le repère de l'image affichée (après rotation). */
data class Detection(
    val cls: Int,
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

data class DetectionResult(
    val detections: List<Detection>,
    val frameW: Int,
    val frameH: Int,
    val preMs: Float,
    val inferMs: Float
)

/**
 * Détecteur YOLO26 (export TFLite, sortie end-to-end [1, N, 6] = x1,y1,x2,y2,score,classe).
 * Toutes les méthodes doivent être appelées depuis le MÊME thread (exigence du délégué GPU).
 */
class YoloDetector(
    context: Context,
    assetName: String,
    private val gpuSupported: Boolean
) {
    companion object {
        /** Si les boîtes sont décalées/aberrantes, essayez BoxFormat.XYWH (voir README). */
        val BOX_FORMAT = BoxFormat.XYXY
    }

    private val modelBuffer: ByteBuffer = context.assets.open(assetName).use { ins ->
        val bytes = ins.readBytes()
        ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
            put(bytes)
            rewind()
        }
    }

    private var interpreter: InterpreterApi? = null
    private var lastWantGpu: Boolean? = null

    @Volatile
    var backendLabel: String = "-"
        private set

    @Volatile
    var inputLabel: String = "-"
        private set

    // --- Entrée ---
    private var inW = 0
    private var inH = 0
    private var inType = DataType.FLOAT32
    private var inScale = 0f
    private var inZero = 0
    private var inChannelFirst = false   // true : entrée NCHW [1,3,H,W]
    private lateinit var inBuf: ByteBuffer

    // --- Sortie ---
    private var outRows = 0
    private var outCols = 0            // nb de valeurs par détection (6 en end-to-end, 4+nb classes en brut)
    private var rawChannelFirst = false // true : sortie brute [1, 4+nc, N] (export end2end=False)
    private var outType = DataType.FLOAT32
    private var outScale = 0f
    private var outZero = 0
    private lateinit var outBuf: ByteBuffer
    private var outValues = FloatArray(0)

    // --- Prétraitement ---
    private lateinit var letter: Bitmap
    private lateinit var letterCanvas: Canvas
    private var pixels = IntArray(0)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private var loggedOnce = false

    /** Crée (ou recrée) l'interpréteur si le backend demandé a changé. */
    fun ensureBackend(wantGpu: Boolean) {
        if (interpreter != null && lastWantGpu == wantGpu) return
        close()
        lastWantGpu = wantGpu
        loggedOnce = false

        if (wantGpu && gpuSupported) {
            var gpuInterp: InterpreterApi? = null
            try {
                gpuInterp = build(true)
                setupTensors(gpuInterp)
                warmup(gpuInterp)
                interpreter = gpuInterp
                backendLabel = "GPU"
                return
            } catch (e: Throwable) {
                Log.w(TAG, "GPU inutilisable, repli sur CPU", e)
                try { gpuInterp?.close() } catch (_: Throwable) {}
            }
        }
        val cpu = build(false)
        setupTensors(cpu)
        warmup(cpu)
        interpreter = cpu
        backendLabel = if (wantGpu) "CPU (repli : GPU non utilisable)" else "CPU"
    }

    private fun build(gpu: Boolean): InterpreterApi {
        val opts = InterpreterApi.Options()
            .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
        if (gpu) opts.addDelegateFactory(GpuDelegateFactory()) else opts.setNumThreads(4)
        modelBuffer.rewind()
        return InterpreterApi.create(modelBuffer, opts)
    }

    private fun setupTensors(it: InterpreterApi) {
        val inT = it.getInputTensor(0)
        val inShape = inT.shape()
        require(inShape.size == 4 && (inShape[3] == 3 || inShape[1] == 3)) {
            "Entrée inattendue ${inShape.contentToString()} (attendu NHWC [1,H,W,3] ou NCHW [1,3,H,W])"
        }
        inChannelFirst = inShape[3] != 3
        if (inChannelFirst) {
            inH = inShape[2]
            inW = inShape[3]
        } else {
            inH = inShape[1]
            inW = inShape[2]
        }
        inType = inT.dataType()
        inScale = inT.quantizationParams().scale
        inZero = inT.quantizationParams().zeroPoint
        inBuf = ByteBuffer.allocateDirect(inT.numBytes()).order(ByteOrder.nativeOrder())

        val outT = it.getOutputTensor(0)
        val outShape = outT.shape()
        require(outShape.size == 3 && outShape[1] >= 5 && outShape[2] >= 5) {
            "Sortie inattendue ${outShape.contentToString()} : attendu [1,N,6] ou [1,4+nc,N]"
        }
        // [1,6,8400] (brut, channel-first) vs [1,300,6] (end-to-end)
        rawChannelFirst = outShape[1] < outShape[2]
        if (rawChannelFirst) {
            outCols = outShape[1]
            outRows = outShape[2]
        } else {
            outRows = outShape[1]
            outCols = outShape[2]
        }
        outType = outT.dataType()
        outScale = outT.quantizationParams().scale
        outZero = outT.quantizationParams().zeroPoint
        outBuf = ByteBuffer.allocateDirect(outT.numBytes()).order(ByteOrder.nativeOrder())
        outValues = FloatArray(outRows * outCols)

        letter = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888)
        letterCanvas = Canvas(letter)
        pixels = IntArray(inW * inH)
        inputLabel = "${inW}x$inH ${inType.name.lowercase()}${if (inChannelFirst) " NCHW" else ""}"
        Log.d(TAG, "Décodage : ${if (rawChannelFirst) "brut channel-first + NMS" else "end-to-end"}")
        Log.d(TAG, "Entrée ${inShape.contentToString()} $inType (scale=$inScale zp=$inZero) | " +
            "Sortie ${outShape.contentToString()} $outType (scale=$outScale zp=$outZero)")
    }

    private fun warmup(it: InterpreterApi) {
        inBuf.rewind()
        outBuf.rewind()
        it.run(inBuf, outBuf)
    }

    fun detect(src: Bitmap, rotationDeg: Int, confThr: Float): DetectionResult {
        val interp = interpreter ?: error("Interpréteur non initialisé")
        val t0 = System.nanoTime()

        // Image affichée = image capteur tournée de rotationDeg
        val sw = src.width
        val sh = src.height
        val swapped = rotationDeg % 180 != 0
        val fw = if (swapped) sh else sw
        val fh = if (swapped) sw else sh

        // Letterbox (fond gris 114 comme à l'entraînement), rotation incluse
        val scale = min(inW.toFloat() / fw, inH.toFloat() / fh)
        val padX = (inW - fw * scale) / 2f
        val padY = (inH - fh * scale) / 2f
        matrix.reset()
        matrix.postTranslate(-sw / 2f, -sh / 2f)
        matrix.postRotate(rotationDeg.toFloat())
        matrix.postScale(scale, scale)
        matrix.postTranslate(inW / 2f, inH / 2f)
        letterCanvas.drawColor(Color.rgb(114, 114, 114))
        letterCanvas.drawBitmap(src, matrix, paint)
        letter.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        fillInput()
        val t1 = System.nanoTime()

        inBuf.rewind()
        outBuf.rewind()
        interp.run(inBuf, outBuf)
        val t2 = System.nanoTime()

        val dets = decode(confThr, fw, fh, scale, padX, padY)
        return DetectionResult(dets, fw, fh, (t1 - t0) / 1e6f, (t2 - t1) / 1e6f)
    }

    private fun fillInput() {
        inBuf.rewind()
        when (inType) {
            DataType.FLOAT32 -> {
                val fb = inBuf.asFloatBuffer()
                val inv = 1f / 255f
                if (inChannelFirst) {
                    for (shift in intArrayOf(16, 8, 0))
                        for (p in pixels) fb.put(((p shr shift) and 0xFF) * inv)
                } else {
                    for (p in pixels) {
                        fb.put(((p shr 16) and 0xFF) * inv)
                        fb.put(((p shr 8) and 0xFF) * inv)
                        fb.put((p and 0xFF) * inv)
                    }
                }
            }
            DataType.UINT8, DataType.INT8 -> {
                val signed = inType == DataType.INT8
                if (inChannelFirst) {
                    for (shift in intArrayOf(16, 8, 0))
                        for (p in pixels) inBuf.put(quantize((p shr shift) and 0xFF, signed))
                } else {
                    for (p in pixels) {
                        inBuf.put(quantize((p shr 16) and 0xFF, signed))
                        inBuf.put(quantize((p shr 8) and 0xFF, signed))
                        inBuf.put(quantize(p and 0xFF, signed))
                    }
                }
            }
            else -> error("Type d'entrée non géré : $inType")
        }
        inBuf.rewind()
    }

    private fun quantize(raw: Int, signed: Boolean): Byte {
        val lo = if (signed) -128 else 0
        val hi = if (signed) 127 else 255
        val v = if (inScale == 0f) {
            raw + lo                                   // pas de quantification déclarée
        } else {
            ((raw / 255f) / inScale + inZero).roundToInt().coerceIn(lo, hi)
        }
        return v.toByte()
    }

    private fun readOutput() {
        outBuf.rewind()
        when (outType) {
            DataType.FLOAT32 -> outBuf.asFloatBuffer().get(outValues)
            DataType.INT8 -> for (i in outValues.indices) {
                outValues[i] = (outBuf.get(i) - outZero) * outScale
            }
            DataType.UINT8 -> for (i in outValues.indices) {
                outValues[i] = ((outBuf.get(i).toInt() and 0xFF) - outZero) * outScale
            }
            else -> error("Type de sortie non géré : $outType")
        }
    }

    private fun decode(thr: Float, fw: Int, fh: Int, scale: Float, padX: Float, padY: Float): List<Detection> {
        readOutput()
        return if (rawChannelFirst) decodeRaw(thr, fw, fh, scale, padX, padY)
        else decodeEndToEnd(thr, fw, fh, scale, padX, padY)
    }

    /** Sortie brute [1, 4+nc, N] : cx,cy,w,h puis un score par classe. NMS appliqué ici. */
    private fun decodeRaw(thr: Float, fw: Int, fh: Int, scale: Float, padX: Float, padY: Float): List<Detection> {
        val v = outValues
        val n = outRows
        val nc = outCols - 4

        // 1) candidats au-dessus du seuil : [cx, cy, w, h, score, classe] (repère letterbox, pixels)
        val cand = ArrayList<FloatArray>()
        var maxCoord = 0f
        for (i in 0 until n) {
            var best = -1
            var bestScore = thr
            for (c in 0 until nc) {
                val sc = v[(4 + c) * n + i]
                if (sc >= bestScore) { bestScore = sc; best = c }
            }
            if (best < 0) continue
            val box = FloatArray(6)
            for (k in 0..3) {
                box[k] = v[k * n + i]
                maxCoord = max(maxCoord, box[k])
            }
            box[4] = bestScore
            box[5] = best.toFloat()
            cand.add(box)
        }
        if (cand.isEmpty()) return emptyList()

        val normalized = maxCoord <= 1.5f
        val sx = if (normalized) inW.toFloat() else 1f
        val sy = if (normalized) inH.toFloat() else 1f
        for (b in cand) {
            b[0] *= sx; b[1] *= sy; b[2] *= sx; b[3] *= sy
        }

        // 2) NMS par classe
        cand.sortByDescending { it[4] }
        val kept = ArrayList<FloatArray>()
        for (b in cand) {
            if (kept.none { it[5] == b[5] && iou(it, b) > 0.45f }) kept.add(b)
            if (kept.size >= 50) break
        }

        // 3) repère letterbox -> image affichée normalisée
        val res = ArrayList<Detection>()
        for (b in kept) {
            val l = ((b[0] - b[2] / 2f - padX) / scale / fw).coerceIn(0f, 1f)
            val t = ((b[1] - b[3] / 2f - padY) / scale / fh).coerceIn(0f, 1f)
            val r = ((b[0] + b[2] / 2f - padX) / scale / fw).coerceIn(0f, 1f)
            val bt = ((b[1] + b[3] / 2f - padY) / scale / fh).coerceIn(0f, 1f)
            if (r - l < 0.005f || bt - t < 0.005f) continue
            if (!loggedOnce) {
                loggedOnce = true
                Log.d(TAG, "1re détection brute : ${b.joinToString()} (normalisé=$normalized)")
            }
            res.add(Detection(b[5].toInt(), b[4], l, t, r, bt))
        }
        return res
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val ax1 = a[0] - a[2] / 2f; val ay1 = a[1] - a[3] / 2f
        val ax2 = a[0] + a[2] / 2f; val ay2 = a[1] + a[3] / 2f
        val bx1 = b[0] - b[2] / 2f; val by1 = b[1] - b[3] / 2f
        val bx2 = b[0] + b[2] / 2f; val by2 = b[1] + b[3] / 2f
        val iw = min(ax2, bx2) - max(ax1, bx1)
        val ih = min(ay2, by2) - max(ay1, by1)
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        return inter / (a[2] * a[3] + b[2] * b[3] - inter)
    }

    /** Sortie end-to-end [1, N, 6] : x1,y1,x2,y2,score,classe. */
    private fun decodeEndToEnd(thr: Float, fw: Int, fh: Int, scale: Float, padX: Float, padY: Float): List<Detection> {
        val v = outValues

        // Coordonnées normalisées (0..1) ou en pixels (0..640) ? On le détecte.
        var maxCoord = 0f
        for (i in 0 until outRows) {
            val o = i * 6
            if (v[o + 4] >= thr) for (k in 0..3) maxCoord = max(maxCoord, v[o + k])
        }
        val normalized = maxCoord > 0f && maxCoord <= 1.5f
        val sx = if (normalized) inW.toFloat() else 1f
        val sy = if (normalized) inH.toFloat() else 1f

        val res = ArrayList<Detection>()
        for (i in 0 until outRows) {
            val o = i * 6
            val score = v[o + 4]
            if (score < thr) continue
            val a = v[o] * sx
            val b = v[o + 1] * sy
            val c = v[o + 2] * sx
            val d = v[o + 3] * sy
            val xyxy = BOX_FORMAT == BoxFormat.XYXY
            val x1 = if (xyxy) a else a - c / 2f
            val y1 = if (xyxy) b else b - d / 2f
            val x2 = if (xyxy) c else a + c / 2f
            val y2 = if (xyxy) d else b + d / 2f

            // repère letterbox -> repère image affichée, normalisé 0..1
            val l = ((x1 - padX) / scale / fw).coerceIn(0f, 1f)
            val t = ((y1 - padY) / scale / fh).coerceIn(0f, 1f)
            val r = ((x2 - padX) / scale / fw).coerceIn(0f, 1f)
            val bt = ((y2 - padY) / scale / fh).coerceIn(0f, 1f)
            if (r - l < 0.005f || bt - t < 0.005f) continue

            if (!loggedOnce) {
                loggedOnce = true
                Log.d(TAG, "1re ligne brute de sortie : ${v.copyOfRange(o, o + 6).joinToString()} " +
                    "(normalisé=$normalized, format=$BOX_FORMAT)")
            }
            res.add(Detection(v[o + 5].roundToInt(), score, l, t, r, bt))
        }
        return res
    }

    fun close() {
        try { interpreter?.close() } catch (_: Throwable) {}
        interpreter = null
    }
}
