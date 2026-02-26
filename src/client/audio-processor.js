/**
 * VoiceProcessor — AudioWorkletProcessor for voice capture.
 *
 * Runs on a dedicated audio thread (not the main thread).
 * Accumulates 128-sample frames into 4096-sample chunks,
 * computes per-chunk energy, and posts to the main thread.
 */
class VoiceProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Float32Array(4096);
        this._writeIndex = 0;
        this._active = true;

        this.port.onmessage = (e) => {
            if (e.data.type === 'stop') this._active = false;
            if (e.data.type === 'start') { this._active = true; this._writeIndex = 0; }
        };
    }

    process(inputs) {
        if (!this._active) return true;

        const input = inputs[0];
        if (!input || !input[0]) return true;

        const samples = input[0]; // mono channel, 128 samples per call (at 16kHz render quantum)

        for (let i = 0; i < samples.length; i++) {
            this._buffer[this._writeIndex++] = samples[i];

            if (this._writeIndex >= 4096) {
                // Compute energy for this chunk
                let energy = 0;
                for (let j = 0; j < 4096; j++) energy += Math.abs(this._buffer[j]);
                energy /= 4096;

                // Transfer buffer to main thread (zero-copy)
                const copy = this._buffer.slice();
                this.port.postMessage(
                    { type: 'audio', buffer: copy, energy },
                    [copy.buffer]
                );
                this._writeIndex = 0;
            }
        }

        return true; // keep processor alive
    }
}

registerProcessor('voice-processor', VoiceProcessor);
