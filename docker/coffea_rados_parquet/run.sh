#!/bin/bash
set -ex

function fail {
  echo $1 >&2
  exit 1
}

function retry {
  local n=1
  local max=3
  local delay=5
  while true; do
    "$@" && break || {
      if [[ $n -lt $max ]]; then
        ((n++))
        echo "Command failed. Attempt $n/$max:"
        sleep $delay;
      else
        fail "The command has failed after $n attempts."
      fi
    }
  done
}

retry docker run \
        -v $(pwd):/w \
        -w /w \
        -e IS_CI=true \
        --privileged \
        coffea-rados-parquet-test \
        ./docker/coffea_rados_parquet/script.sh

