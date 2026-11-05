# FPV Footage to Video

Desktop app that scans a folder (microSD, USB drive, etc.) of FPV drone footage, keeps clips with at least **10 seconds of real flight**, scores speed, writes AI editing notes, and copies keepers to a folder on your PC.

## Launch the app

Double-click:

```
Launch FPV Clip Triage.bat
```

Or from a terminal:

```powershell
cd C:\Users\awitt\fpv-clip-triage
.\.venv\Scripts\Activate.ps1
python -m fpv_triage
```

1. Browse to your SD card / footage folder  
2. Choose an output folder on the PC (default: `Videos\fpv-keepers`)  
3. Leave **AI descriptions** on to write a `.txt` notes file next to each keeper  
4. Optional: enable **Dry run** to preview without copying video files (notes still write)  
5. Click **Start triage**

First AI run downloads **Qwen3-VL-4B-Instruct** to your Hugging Face cache (one-time).

## How it decides

1. Skips files shorter than the minimum flight length.
2. Samples frames (~2/sec) and measures optical-flow motion.
3. Marks sustained high-motion stretches as flight.
4. Keeps the file if total flight time is at least **Min flight (sec)** (default: 10).
5. Scores **overall speed** and **top speed** (1–10) from that motion.
6. For keepers, a local vision model describes the clip for later editing.
7. Copies keepers into the output folder, writes matching `.txt` notes, and `triage_report.json`.

Tune **Motion threshold** if your camera is noisier or smoother than typical FPV. Higher = stricter.

### Notes file example

```
File: DJI_0023.mp4
Duration: 48.2s
Flight time: 41.0s
Overall speed score: 6.4 / 10
Top speed score: 8.1 / 10

Description:
FPV flight through a tree line at dusk...
```

## Command line (optional)

```powershell
python -m fpv_triage --cli --source E:\DCIM --output C:\Users\awitt\Videos\fpv-keepers
python -m fpv_triage --cli --source E:\DCIM --output C:\Users\awitt\Videos\fpv-keepers --dry-run
```
