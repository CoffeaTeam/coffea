FROM uccross/skyhookdm-arrow:v0.4.0

RUN yum -y update && \
    yum -y install python3 \
                   python3-devel \
                   python3-pip \
                   llvm-devel && \
    pip3 install pip --upgrade
