package com.example.falldetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/** Dessine les boîtes au-dessus de l'aperçu caméra (PreviewView en mode fitCenter). */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    companion object {
        val LABELS = arrayOf("fall", "non-fall")
        private val COLORS = intArrayOf(Color.rgb(229, 57, 53), Color.rgb(67, 160, 71))
    }

    private var detections: List<Detection> = emptyList()
    private var frameW = 0
    private var frameH = 0

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 38f
        typeface = Typeface.DEFAULT_BOLD
    }

    fun setResults(dets: List<Detection>, fw: Int, fh: Int) {
        detections = dets
        frameW = fw
        frameH = fh
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (frameW == 0 || frameH == 0) return

        // zone réellement occupée par l'image (fitCenter)
        val s = min(width / frameW.toFloat(), height / frameH.toFloat())
        val dw = frameW * s
        val dh = frameH * s
        val ox = (width - dw) / 2f
        val oy = (height - dh) / 2f

        for (d in detections) {
            val color = COLORS[d.cls.coerceIn(0, COLORS.size - 1)]
            val l = ox + d.left * dw
            val t = oy + d.top * dh
            val r = ox + d.right * dw
            val b = oy + d.bottom * dh
            boxPaint.color = color
            canvas.drawRect(l, t, r, b, boxPaint)

            val name = LABELS.getOrElse(d.cls) { "cls${d.cls}" }
            val text = "$name ${(d.score * 100).toInt()}%"
            val tw = textPaint.measureText(text)
            fillPaint.color = color
            canvas.drawRect(l, t - 46f, l + tw + 16f, t, fillPaint)
            canvas.drawText(text, l + 8f, t - 10f, textPaint)
        }
    }
}
