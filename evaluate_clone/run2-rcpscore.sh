# INPUT_DIR="./data-py"
# Java
RCP_SCR_DIR="./output-java"
MODELS="codegpt,codebert,codet5,graphcodebert"
PRG="4_plot_boxplot_rcpscore.py"
# Python
RCP_SCR_DIR="./output-py"
# 

echo " "
echo "========================================="
echo "== Generating Delta Charts"
echo "========================================="

# 2. Convert comma-separated string to an array and loop to generate charts
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

for MODEL in "${MODEL_ARRAY[@]}"; do
    JSONL_FILE="${RCP_SCR_DIR}/${MODEL}/rcp_scores.jsonl"
    # DELTA_CHART="rcp_delta_chart_${MODEL}.png"
    BOX_PLOT="rcp_boxplot_${MODEL}.png"
    
    if [ -f "$JSONL_FILE" ]; then
        echo "Generating charts for $MODEL..."
        # python 2_plot_delta.py --rcp "$JSONL_FILE" --output "$DELTA_CHART"
        python ${PRG} --rcp "$JSONL_FILE" --output "$BOX_PLOT"
    fi
done

echo " "
echo "** Pipeline finished! All reports and charts are in $RCP_SCR_DIR"