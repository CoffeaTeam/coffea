#!/bin/bash
set -e
set -x
set -u

DIR=/tmp/test-cluster

# reset
pkill ceph || true
rm -rf ${DIR}/*
LOG_DIR=${DIR}/log
MON_DATA=${DIR}/mon
MDS_DATA=${DIR}/mds
MOUNTPT=${MDS_DATA}/mnt
OSD_DATA=${DIR}/osd
mkdir ${LOG_DIR} ${MON_DATA} ${OSD_DATA} ${MDS_DATA} ${MOUNTPT}
MDS_NAME="Z"
MON_NAME="a"
MGR_NAME="x"

# cluster wide parameters
cat >> ${DIR}/ceph.conf <<EOF
[global]
fsid = $(uuidgen)
osd crush chooseleaf type = 0
run dir = ${DIR}/run
auth cluster required = none
auth service required = none
auth client required = none
osd pool default size = 1
[mds.${MDS_NAME}]
host = localhost
[mon.${MON_NAME}]
log file = ${LOG_DIR}/mon.log
chdir = ""
mon cluster log file = ${LOG_DIR}/mon-cluster.log
mon data = ${MON_DATA}
mon data avail crit = 0
mon addr = 127.0.0.1
mon allow pool delete = true
[osd.0]
log file = ${LOG_DIR}/osd.log
chdir = ""
osd data = ${OSD_DATA}
osd journal = ${OSD_DATA}.journal
osd journal size = 100
osd objectstore = memstore
osd class load list = *
osd class default list = *
EOF

export CEPH_CONF=${DIR}/ceph.conf

# start an osd
ceph-mon --id ${MON_NAME} --mkfs --keyring /dev/null
touch ${MON_DATA}/keyring
ceph-mon --id ${MON_NAME}

# start an osd
OSD_ID=$(ceph osd create)
ceph osd crush add osd.${OSD_ID} 1 root=default host=localhost
ceph-osd --id ${OSD_ID} --mkjournal --mkfs
sleep 5 # this is an attempt to fix CI issue #423, remove if it has no effect
ceph-osd --id ${OSD_ID}

# start an mds for cephfs
ceph auth get-or-create mds.${MDS_NAME} mon 'profile mds' mgr 'profile mds' mds 'allow *' osd 'allow *' > ${MDS_DATA}/keyring
ceph osd pool create cephfs_data 8
ceph osd pool create cephfs_metadata 8
ceph fs new cephfs cephfs_metadata cephfs_data
ceph fs ls
ceph-mds -i ${MDS_NAME}
ceph status
while [[ ! $(ceph mds stat | grep "up:active") ]]; do sleep 1; done


# start a manager
ceph-mgr --id ${MGR_NAME}

# test the setup
ceph --version
ceph status

# mount cephfs
mkdir -p /mnt/cephfs
ceph-fuse --id client.admin -m 127.0.0.1:6789  --client_fs cephfs /mnt/cephfs
sleep 5

# download an example dataset and copy into the mounted dir
rm -rf nyc*
wget https://raw.githubusercontent.com/JayjeetAtGithub/zips/main/nyc.zip # try to get this dataset into the source tree
unzip nyc.zip
cp -r nyc /mnt/cephfs/
sleep 15

# install required dependencies
pip3 install dask[distributed] nbconvert
pip3 install 'fsspec>=0.3.3'
pip3 install --upgrade . # install coffea
pip3 install --upgrade /pyarrow-*.whl # update the PyArrow with Rados parquet extensions


if [ ! -z "$IS_CI" ]; then
    python3 tests/test_rados_parquet_job.py
else
    # start the notebook
    jupyter notebook --allow-root --no-browser  --ip 0.0.0.0
fi

# unmount cephfs
umount /mnt/cephfs
