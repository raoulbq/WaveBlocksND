#! /bin/bash

SRCPATH="$HOME/WaveBlocksND/"

if [ $# -lt 1 ]
    then
    IPATH="$HOME/wbi";
    echo "Need an installation path, None given, using default";
else
    IPATH=$1;
fi

echo "Install WB to location $IPATH";

# Create dir if necessary
if [ ! -d $IPATH ]
    then
    mkdir -p $IPATH
fi

# Copy scripts
cp $SRCPATH/src/scripts/*.py $IPATH/
cp $SRCPATH/src/plotters/*.py $IPATH/
