package com.example.falldetect

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        const val MODEL_ASSET = "model.tflite"
        const val FALL_CLASS = 0          // names: {0: 'fall', 1: 'non-fall'}
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var statsText: TextView
    private lateinit var alertBanner: TextView
    private lateinit var thresholdLabel: TextView
    private lateinit var thresholdSeek: SeekBar
    private lateinit var backendButton: Button

    private var detector: YoloDetector? = null
    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val alarm = FallAlarm()

    @Volatile private var useGpu = true
    @Volatile private var threshold = 0.40f

    private var gpuAvailable = false
    private var permissionOk = false
    private var runtimeReady = false
    private var cameraStarted = false

    // statistiques (modifiées uniquement sur le thread d'inférence)
    private var emaInfer = 0f
    private var emaPre = 0f
    private var emaFps = 0f
    private var lastFrameNs = 0L

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                permissionOk = true
                startCameraIfReady()
            } else {
                Toast.makeText(this, "Permission caméra refusée", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)
        statsText = findViewById(R.id.statsText)
        alertBanner = findViewById(R.id.alertBanner)
        thresholdLabel = findViewById(R.id.thresholdLabel)
        thresholdSeek = findViewById(R.id.thresholdSeek)
        backendButton = findViewById(R.id.backendButton)

        thresholdSeek.progress = 30                      // 0,40 par défaut
        updateThresholdLabel()
        thresholdSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                threshold = (p + 10) / 100f
                updateThresholdLabel()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        updateBackendButton()
        backendButton.setOnClickListener {
            useGpu = !useGpu
            updateBackendButton()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            permissionOk = true
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
        initRuntime()
    }

    /** Initialise LiteRT (runtime fourni par Google Play services) avec support GPU si dispo. */
    private fun initRuntime() {
        TfLiteGpu.isGpuDelegateAvailable(this)
            .onSuccessTask { gpu ->
                gpuAvailable = gpu == true
                TfLite.initialize(
                    this,
                    TfLiteInitializationOptions.builder()
                        .setEnableGpuDelegateSupport(gpuAvailable)
                        .build()
                )
            }
            .addOnSuccessListener {
                try {
                    detector = YoloDetector(applicationContext, MODEL_ASSET, gpuAvailable)
                    runtimeReady = true
                    startCameraIfReady()
                } catch (e: Exception) {
                    Log.e(TAG, "Chargement du modèle impossible", e)
                    statsText.text = "Modèle introuvable.\nCopiez model.tflite dans\napp/src/main/assets/"
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Initialisation LiteRT impossible", e)
                statsText.text = "Init LiteRT impossible :\n${e.message}"
            }
    }

    private fun startCameraIfReady() {
        if (permissionOk && runtimeReady && !cameraStarted) {
            cameraStarted = true
            startCamera()
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            // même ratio 4:3 pour l'aperçu et l'analyse => les boîtes tombent au bon endroit
            val resolution = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()

            val preview = Preview.Builder()
                .setResolutionSelector(resolution)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(inferenceExecutor) { proxy -> analyze(proxy) }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    /** Exécuté sur le thread d'inférence unique. */
    private fun analyze(proxy: ImageProxy) {
        val det = detector
        if (det == null) {
            proxy.close()
            return
        }
        try {
            det.ensureBackend(useGpu)
            val bitmap = proxy.toBitmap()
            val res = det.detect(bitmap, proxy.imageInfo.rotationDegrees, threshold)

            val hit = res.detections.any { it.cls == FALL_CLASS }
            val state = alarm.update(hit, SystemClock.elapsedRealtime())

            val now = System.nanoTime()
            emaInfer = if (emaInfer == 0f) res.inferMs else 0.9f * emaInfer + 0.1f * res.inferMs
            emaPre = if (emaPre == 0f) res.preMs else 0.9f * emaPre + 0.1f * res.preMs
            if (lastFrameNs != 0L) {
                val fps = 1e9f / (now - lastFrameNs)
                emaFps = if (emaFps == 0f) fps else 0.9f * emaFps + 0.1f * fps
            }
            lastFrameNs = now

            val text = String.format(
                Locale.US, "%s | %s\ninférence %.1f ms | prétraitement %.1f ms | %.0f fps",
                det.backendLabel, det.inputLabel, emaInfer, emaPre, emaFps
            )
            runOnUiThread {
                overlay.setResults(res.detections, res.frameW, res.frameH)
                statsText.text = text
                alertBanner.visibility = if (state.active) View.VISIBLE else View.GONE
                if (state.justTriggered) notifyFall()
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Erreur d'inférence", t)
            runOnUiThread { statsText.text = "Erreur :\n${t.message}" }
        } finally {
            proxy.close()
        }
    }

    private fun notifyFall() {
        try {
            val vib = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            vib.vibrate(VibrationEffect.createOneShot(700, VibrationEffect.DEFAULT_AMPLITUDE))
        } catch (_: Throwable) {}
        try {
            ToneGenerator(AudioManager.STREAM_ALARM, 100)
                .startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 800)
        } catch (_: Throwable) {}
    }

    private fun updateThresholdLabel() {
        thresholdLabel.text = String.format(Locale.US, "Seuil de confiance : %.2f", (thresholdSeek.progress + 10) / 100f)
    }

    private fun updateBackendButton() {
        backendButton.text = if (useGpu) "Backend demandé : GPU (toucher pour CPU)" else "Backend demandé : CPU (toucher pour GPU)"
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceExecutor.execute { detector?.close() }
        inferenceExecutor.shutdown()
    }
}
