#!/bin/bash
check() {
    node=$1
    if ssh -o ConnectTimeout=2 -q "$node" df -h | grep pacific ; then
        echo -e "${a}: ${node} is OK"
    else
        echo -e "${node} NOK"
    fi
}

while read -r node; do
    # echo $node
    [ -z "$node" ] && continue
    check "$node" &
    # if ! ping -c 1 -W 1 "$node" &> /dev/null; then
    #     echo "$node"
    # fi
done < "$1"


