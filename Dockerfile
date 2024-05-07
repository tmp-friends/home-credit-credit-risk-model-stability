FROM gcr.io/kaggle-gpu-images/python:v137

USER root
ENV TZ=Asia/Tokyo \
    LANG=ja_JP.UTF-8

ARG uid
ARG gid
ARG user

RUN mkdir -p /kaggle \
    && chown -R $user:$user /kaggle \
    && chmod -R 0755 /kaggle \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

WORKDIR /kaggle

#各々のGPUに対応するpytorchをインストールhttps://pytorch.org/get-started/previous-versions/
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
ADD requirements.txt /kaggle/requirements.txt
RUN pip install -r requirements.txt

#jupyter notebookの起動
ADD run.sh /opt/run.sh
RUN chown $user:$user /opt/run.sh \
    && chmod 0755 /opt/run.sh

USER $user
CMD /opt/run.sh
