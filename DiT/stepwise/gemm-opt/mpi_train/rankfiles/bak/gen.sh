#!/bin/bash
hosts=("29.204.25.50" "29.204.25.51" "29.204.25.52" "29.204.25.53" "29.204.25.54" "29.204.25.55" "29.204.25.56" "29.204.25.57")

hnum=0

read -p "input rankfile name:  " rankfile

echo "" > $rankfile


while true; do
	read -p "input host${hnum}:  " host
	if [[ "$host" == "-1" ]]; then
		break
	fi
	for ((i = 0;i<4;i++)); do
		if (( i % 2 == 0 ));then
			echo "rank $((i+4*hnum)) = 29.204.25.$host slot=$((i/2)):0-37,$((i/2)):39-76,$((i/2)):77-114,$((i/2)):115-152" >> $rankfile
		else
	
			echo "rank $((i+4*hnum)) = 29.204.25.$host slot=$((i/2)):153-190, $((i/2)):192-229,$((i/2)):230-267, $((i/2)):268-305" >> $rankfile
		fi
	done
	hnum=$((hnum+1))
done

