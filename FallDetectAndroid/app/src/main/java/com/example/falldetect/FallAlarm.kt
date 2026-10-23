package com.example.falldetect

/**
 * Une chute est un événement dans le temps : l'alerte se déclenche si la classe "fall"
 * est vue dans au moins [minHits] des [windowSize] dernières images.
 * Un délai de [cooldownMs] évite de répéter la notification.
 */
class FallAlarm(
    private val windowSize: Int = 8,
    private val minHits: Int = 5,
    private val cooldownMs: Long = 10_000
) {
    data class State(val active: Boolean, val justTriggered: Boolean)

    private val window = BooleanArray(windowSize)
    private var idx = 0
    private var wasActive = false
    private var lastTriggerMs = Long.MIN_VALUE / 2

    fun update(hit: Boolean, nowMs: Long): State {
        window[idx] = hit
        idx = (idx + 1) % windowSize
        val active = window.count { it } >= minHits
        val trigger = active && !wasActive && (nowMs - lastTriggerMs >= cooldownMs)
        if (trigger) lastTriggerMs = nowMs
        wasActive = active
        return State(active, trigger)
    }
}
