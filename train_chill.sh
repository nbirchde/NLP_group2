#!/bin/bash
# Chill Mode Training Launcher
# Runs training with reduced resource usage so you can use your Mac comfortably

echo "🧊 Starting Chill Mode Training..."
echo ""
echo "Config: batch_size=8 (50% of max), nice priority=15"
echo "Expected time: ~20-25 minutes for 5 epochs"
echo "Your Mac will stay cool and usable!"
echo ""
echo "═══════════════════════════════════════════════"

cd /Users/nicholasbirchdelacalle/Documents/NL/NLP_project/NLP_group2

# Run training with low priority in background
nohup nice -n 15 .venv/bin/python experiments/distilbert_text_only/train.py \
  --config configs/chill_mode.yaml \
  > experiments/distilbert_text_only/chill_training.log 2>&1 &

TRAIN_PID=$!

echo "✓ Training started (PID: $TRAIN_PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f experiments/distilbert_text_only/chill_training.log"
echo ""
echo "Check if still running:"
echo "  ps -p $TRAIN_PID"
echo ""
echo "Stop training:"
echo "  kill $TRAIN_PID"
echo ""
echo "═══════════════════════════════════════════════"
echo "You can now:"
echo "  • Browse the web"
echo "  • Write code"  
echo "  • Listen to music"
echo "  • Do whatever!"
echo ""
echo "Training will complete automatically in ~20-25 min."
echo "Results will be in experiments/distilbert_text_only/artifacts/"
