# Load Python Alpine
# Use Alpine due to smaller size compared to Ubuntu and hence giving us a bit more control
# Relies on Python 3
FROM python:alpine

# Our library relies on FFMpeg for a few purposes
RUN apk add --no-cache --virtual .build-deps build-base wget git python3-dev freetype-dev libpng-dev openblas-dev && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
    pip3 install --no-cache-dir numpy==1.15.3
WORKDIR /App

COPY ./App/requirements.txt /App
# Install All Libraries we rely on
RUN     pip3 install --no-cache-dir -r /App/requirements.txt && \
    apk del .build-deps && \
    apk add --no-cache ffmpeg python3 git bash && \
    apk add --no-cache --virtual scipy-runtime libgfortran libgcc libstdc++ musl openblas && \
    rm -rf /var/cache/apk/*

ADD ./App /App

ENTRYPOINT ["python3"]
CMD ["predict.py"]