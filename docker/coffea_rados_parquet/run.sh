#!/bin/bash
set -ex

n=0
until [ "$n" -ge 3 ]
do
   docker run \
        -v $(pwd):/w \
        -w /w \
        -e IS_CI=true \
        --privileged \
        coffea-rados-parquet-test \
        ./docker/coffea_rados_parquet/script.sh && break
   n=$((n+1)) 
   sleep 5
done
