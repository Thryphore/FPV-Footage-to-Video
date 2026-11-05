# FPV Footage to Video

Desktop app that scans a folder (microSD, USB drive, etc.) of FPV drone footage, keeps clips once **5 seconds of flight** are proven with **15 seconds still remaining**, scores speed, writes AI editing notes, and copies keepers to a folder on your PC.

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
4. Keeps the file as soon as **5s of flight** is proven and **≥15s of video remains** after that point (then stops scanning).
5. Scores **overall speed** and **top speed** (1–10) from that motion.
6. For keepers, a local vision model describes the clip for later editing.
7. Copies keepers into the output folder, writes matching `.txt` notes, and `triage_report.json`.

Tune **Motion threshold** if your camera is noisier or smoother than typical FPV. Higher = stricter.

### Notes file example

```
File: DJI_0023.mp4
Duration: 48.2s
Overall speed score: 6.4 / 10
Top speed score: 8.1 / 10
Environment: tree line park

Movement:
Slow cruise at canopy height, yaw left around a trunk, then a short dive toward grass.
```

## AI search

After triage with descriptions on, the output folder gets:

- `*.txt` human notes  
- `*.json` per-clip machine notes  
- `clips_catalog.json` all keepers in one file  
- `clips_search_index.npz` local semantic index  

Use the **AI Search** tab, or:

```powershell
python -m fpv_triage --cli --output C:\Users\awitt\Videos\fpv-keepers --rebuild-index
python -m fpv_triage --cli --output C:\Users\awitt\Videos\fpv-keepers --search "slow cruise near trees"
```

## Command line (optional)

```powershell
python -m fpv_triage --cli --source E:\DCIM --output C:\Users\awitt\Videos\fpv-keepers
python -m fpv_triage --cli --source E:\DCIM --output C:\Users\awitt\Videos\fpv-keepers --dry-run
```
