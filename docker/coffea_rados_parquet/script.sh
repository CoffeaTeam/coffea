#!/bin/bash
set -ex

build_dir=/tmp
test_dir=${build_dir}/test-cluster

pushd ${build_dir}

    # get rid of process and directories leftovers
    pkill ceph-mon || true
    pkill ceph-osd || true
    rm -fr ${test_dir}

    # cluster wide parameters
    mkdir -p ${test_dir}/log
    cat >> /etc/ceph/ceph.conf <<EOF
[global]
fsid = $(uuidgen)
osd crush chooseleaf type = 0
run dir = ${test_dir}/run
auth cluster required = none
auth service required = none
auth client required = none
osd pool default size = 1
EOF
    export CEPH_ARGS="--conf /etc/ceph/ceph.conf"

    # start a MON daemon
    MON_DATA=${test_dir}/mon
    mkdir -p $MON_DATA

    cat >> /etc/ceph/ceph.conf <<EOF
[mon.0]
log file = ${test_dir}/log/mon.log
chdir = ""
mon cluster log file = ${test_dir}/log/mon-cluster.log
mon data = ${MON_DATA}
mon addr = 127.0.0.1
# this was added to enable pool deletion within method delete_one_pool_pp()
mon_allow_pool_delete = true
EOF

    ceph-mon --id 0 --mkfs --keyring /dev/null
    touch ${MON_DATA}/keyring
    cp ${MON_DATA}/keyring /etc/ceph/keyring
    ceph-mon --id 0
    sleep 5

    # start a OSD daemon
    OSD_DATA=${test_dir}/osd
    mkdir ${OSD_DATA}

    cat >> /etc/ceph/ceph.conf <<EOF
[osd.0]
log file = ${test_dir}/log/osd.log
chdir = ""
osd data = ${OSD_DATA}
osd journal = ${OSD_DATA}.journal
osd journal size = 100
osd objectstore = memstore
osd class load list = *
EOF

    OSD_ID=$(ceph osd create)
    ceph osd crush add osd.${OSD_ID} 1 root=default host=localhost
    ceph-osd --id ${OSD_ID} --mkjournal --mkfs
    ceph-osd --id ${OSD_ID}
    sleep 5

    # start a MDS daemon
    MDS_DATA=${TEST_DIR}/mds
    mkdir -p $MDS_DATA

    ceph osd pool create cephfs_data 64
    ceph osd pool create cephfs_metadata 64
    sleep 2

    ceph fs new cephfs cephfs_metadata cephfs_data

    ceph-mds --id a
    while [[ ! $(ceph mds stat | grep "up:active") ]]; do sleep 1; done

    # start a MGR daemon
    ceph-mgr --id 0
    sleep 5

    export CEPH_CONF="/etc/ceph/ceph.conf"

    # mount a ceph filesystem to /mnt/cephfs in the user-space using ceph-fuse
    mkdir -p /mnt/cephfs
    ceph-fuse --id client.admin -m 127.0.0.1:6789  --client_fs cephfs /mnt/cephfs
    sleep 5

    # download an example dataset and copy into the mounted dir
    rm -rf nyc*
    wget https://raw.githubusercontent.com/JayjeetAtGithub/zips/main/nyc.zip # try to get this dataset into the source tree
    unzip nyc.zip
    cp -r nyc /mnt/cephfs/
    sleep 15

popd

pip3 install dask[distributed] nbconvert
pip3 install 'fsspec>=0.3.3'
pip3 install --upgrade . pyarrow<4# install coffea
pip3 install --upgrade /pyarrow-*.whl # update the PyArrow with Rados parquet extensions

if [ ! -z "$IS_CI" ]; then
    python3 tests/test_rados_parquet_job.py
else
    # start the notebook
    jupyter notebook --allow-root --no-browser  --ip 0.0.0.0
fi

# unmount cephfs
umount /mnt/cephfs
