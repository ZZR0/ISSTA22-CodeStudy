INPUT_DIR="./data"
OUTPUT_DIR="./output"
MODELS="codegpt,codebert,codet5,graphcodebert"

echo " "
echo "========================================="
echo "== Generating Delta Charts"
echo "========================================="

# 2. Convert comma-separated string to an array and loop to generate charts
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

for MODEL in "${MODEL_ARRAY[@]}"; do
    JSONL_FILE="${OUTPUT_DIR}/${MODEL}/rcp_scores.jsonl"
    DELTA_CHART="rcp_delta_chart_${MODEL}.png"
    BOX_PLOT="rcp_boxplot_${MODEL}.png"
    
    if [ -f "$JSONL_FILE" ]; then
        echo "Generating charts for $MODEL..."
        python 2_plot_delta.py --rcp "$JSONL_FILE" --output "$DELTA_CHART"
        python 4_plot_boxplot.py --rcp "$JSONL_FILE" --output "$BOX_PLOT"
    fi
done




echo " "
echo "** Pipeline finished! All reports and charts are in $OUTPUT_DIR"