DATA_DIR=$1
CONFIG_FILE=$2
ALLOW_WINDOW_FEATURES=$3
CURR_DIR=$4

python $CURR_DIR/CalculateDHSOverlap.py $DATA_DIR $CONFIG_FILE
python $CURR_DIR/CalculateCSS.py $DATA_DIR $CONFIG_FILE $CURR_DIR

if [ $ALLOW_WINDOW_FEATURES = "1" ];then
	python $CURR_DIR/FindWindows.py $DATA_DIR $CONFIG_FILE
fi

python $CURR_DIR/CalculateFeatures.py $DATA_DIR $CONFIG_FILE $ALLOW_WINDOW_FEATURES
python $CURR_DIR/AggregateFeatures.py $DATA_DIR $CONFIG_FILE $ALLOW_WINDOW_FEATURES
python $CURR_DIR/CombineEPFeatures.py $DATA_DIR $CONFIG_FILE

if [ $ALLOW_WINDOW_FEATURES = "1" ];then
	python $CURR_DIR/AddWindowFeatures.py $DATA_DIR $CONFIG_FILE
fi

