"""Test every available input microphone to find which one actually picks up sound."""
import sounddevice as sd
import numpy as np
import time

print("=" * 60)
print("  MICROPHONE DIAGNOSTIC TEST")
print("=" * 60)
print()
print("This will test EVERY input device on your system.")
print("Please speak loudly or clap while each device is being tested.")
print()

devices = sd.query_devices()
input_devices = []
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        input_devices.append((i, d))

print(f"Found {len(input_devices)} input devices.\n")

results = []

for idx, dev in input_devices:
    name = dev['name']
    sr = int(dev['default_samplerate'])
    channels = min(dev['max_input_channels'], 1)
    
    print(f"  Testing device {idx}: {name}")
    print(f"    Sample rate: {sr} Hz, Channels: {channels}")
    
    try:
        audio = sd.rec(
            int(2 * sr),  # 2 seconds
            samplerate=sr,
            channels=channels,
            dtype='float32',
            device=idx
        )
        sd.wait()
        
        max_vol = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        
        status = "SILENT" if max_vol < 0.005 else "HAS AUDIO!"
        results.append((idx, name, sr, max_vol, rms, status))
        
        print(f"    Max volume: {max_vol:.6f} | RMS: {rms:.6f} | {status}")
        
    except Exception as e:
        print(f"    FAILED: {e}")
        results.append((idx, name, sr, 0, 0, f"ERROR: {e}"))
    
    print()

print("=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print()

working = [(idx, name, sr, mv, rms, st) for idx, name, sr, mv, rms, st in results if st == "HAS AUDIO!"]

if working:
    print(f"WORKING microphones ({len(working)}):")
    for idx, name, sr, mv, rms, st in working:
        print(f"  Device {idx}: {name} (sr={sr}) - max_vol={mv:.4f}")
    
    best = max(working, key=lambda x: x[3])
    print(f"\nBEST device: {best[0]} ({best[1]}) at {best[2]} Hz")
    print(f"\nTo fix audio_listener.py, add this line after sd.rec(...):")
    print(f"  sd.rec(..., device={best[0]})")
else:
    print("NO WORKING MICROPHONES FOUND!")
    print("Your microphone may be disabled in Windows Privacy Settings.")
    print("Go to: Settings > Privacy > Microphone > Allow apps to access your microphone")
