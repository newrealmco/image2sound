# image2sound architecture (v0.1)
Pipeline: image -> features -> music params -> notes -> WAV
Modules:
- features.py: extract brightness/contrast/edge-density + 5-color palette
- mapping.py: map features to BPM, key/scale, instruments, intensity, duration
- compose.py: generate simple arrangement (chords, lead, bass, drums)
- synth.py: render to WAV (additive synth + noise drums)
- cli.py: glue + options (--style, --duration)
Non-goals v0.1: fancy DSP, latency tuning, genre models. Focus on correctness + distinct outputs.
