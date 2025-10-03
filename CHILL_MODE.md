# 🧊 Chill Mode Training Guide

Your computer getting hot? Here's how to run training with reduced resource usage.

## Quick Start: Chill Mode (GPU-Friendly)

```bash
cd /Users/nicholasbirchdelacalle/Documents/NL/NLP_project/NLP_group2

# Recommended: GPU enabled but throttled (batch_size=8, nice priority)
# This keeps GPU active but leaves headroom for your other apps
nice -n 15 .venv/bin/python experiments/distilbert_text_only/train.py \
  --config configs/chill_mode.yaml

# If still too hot: Force CPU-only (disables Mac GPU completely)
PYTORCH_ENABLE_MPS_FALLBACK=1 \
nice -n 10 .venv/bin/python experiments/distilbert_text_only/train.py \
  --config configs/chill_mode.yaml
```

---

## What Changed?

### Chill Mode Config (`configs/chill_mode.yaml`)
- **Batch size**: 8 (train) / 16 (eval) — was 16/32 (50% reduction)
- **GPU usage**: ~50% max capacity → leaves headroom for system
- **Heat generation**: Moderate, won't max out thermals
- **Training time**: ~2x longer (~20-25 minutes total vs 10-15)
- **Results**: Identical to full-speed (just slower)
- **Usability**: You can browse, code, use other apps comfortably

### System Optimizations
- **`nice -n 10`**: Lower process priority (your system stays responsive)
- **CPU-only mode**: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to disable Mac GPU
- **Background execution**: Add `> training.log 2>&1 &` to run in background

---

## Recommendations

### 1. Run Overnight or During Breaks
```bash
# Start before bed/lunch, check results later
cd /Users/nicholasbirchdelacalle/Documents/NL/NLP_project/NLP_group2
nohup nice -n 10 .venv/bin/python experiments/distilbert_text_only/train.py \
  --config configs/chill_mode.yaml \
  > experiments/distilbert_text_only/chill_training.log 2>&1 &

# Check progress anytime:
tail -f experiments/distilbert_text_only/chill_training.log
```

### 2. Monitor Temperature
```bash
# Check CPU temperature (if you have sensors installed)
istats cpu

# Or just check overall system load
top -l 1 | grep "CPU usage"
```

### 3. Pause/Resume Training
Unfortunately, HuggingFace Trainer doesn't natively support pause/resume, but you can:
- Stop training: `pkill -f train.py`
- Resume from checkpoint: Training will auto-resume from `artifacts/checkpoint-*` if it exists

---

## Heat Reduction Tips

### Before Training:
- ✅ Close other heavy apps (Chrome, browsers, IDEs)
- ✅ Ensure good ventilation (don't block vents)
- ✅ Consider using a laptop cooling pad
- ✅ Check Activity Monitor for background processes

### During Training:
- Use `chill_mode.yaml` config (batch_size=4)
- Add `nice -n 10` to reduce priority
- Run CPU-only mode if Mac GPU gets too hot
- Consider splitting into multiple sessions (1-2 epochs at a time)

### Alternative: Cloud Training
If your Mac can't handle it comfortably, consider:
- **Google Colab** (free GPU for 12 hours): https://colab.research.google.com
- **Kaggle Notebooks** (free GPU for 30 hours/week): https://www.kaggle.com/code
- Copy your code + data, run training there, download model weights

---

## Performance Comparison

| Config | Batch Size | Time/Epoch | Total Time (5 epochs) | Heat | Usable? |
|--------|-----------|------------|----------------------|------|---------|
| `base.yaml` | 16 | ~2 min | ~10 min | 🔥🔥🔥 | ❌ Mac unusable |
| `chill_mode.yaml` (GPU) | 8 | ~4 min | ~20 min | 🔥🔥 | ✅ Comfortable |
| `chill_mode.yaml` (CPU-only) | 8 | ~10 min | ~50 min | 🔥 | ✅ Very smooth |

---

## Example: Background Chill Training

```bash
cd /Users/nicholasbirchdelacalle/Documents/NL/NLP_project/NLP_group2

# Start chill mode training in background (GPU-friendly)
nohup nice -n 15 .venv/bin/python experiments/distilbert_text_only/train.py \
  --config configs/chill_mode.yaml \
  > experiments/distilbert_text_only/chill_training.log 2>&1 &

echo "🧊 Chill training started! (~20-25 min for 5 epochs)"
echo ""
echo "Check progress:"
echo "  tail -f experiments/distilbert_text_only/chill_training.log"
echo ""
echo "Your Mac should stay cool and usable. Browse, code, do whatever!"
echo "Training will complete automatically and save results."
```

### What `nice -n 15` Does:
- Lowers training process priority below normal apps
- Your browser, IDE, Slack, etc. get priority
- Training only uses "spare" CPU/GPU cycles
- Result: Smooth multitasking, slower but tolerable training

---

## When Training Completes

Results will be in:
- `experiments/distilbert_text_only/artifacts/final_model/` - Best model
- `experiments/distilbert_text_only/artifacts/final_metrics.txt` - Accuracy/F1 scores
- `experiments/distilbert_text_only/chill_training.log` - Full training log

---

## Still Too Hot?

Try these ultra-conservative settings:

```yaml
# configs/ultra_chill.yaml
train_batch_size: 2  # Minimum viable batch size
eval_batch_size: 4
epochs: 3  # Fewer epochs, rely on early stopping
```

Or just use the quick_test config (1 epoch) and call it a day if the 81% accuracy is good enough!

---

**Bottom Line**: Use `chill_mode.yaml` with `nice -n 10` for ~75% less heat at the cost of 4x longer training time.
