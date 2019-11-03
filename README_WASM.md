# KataGo on Browser
KataGo powered by WebAssembly

## Install
### Convert a weight for TensorFlow.js

```sh
mkdir models
cd models
curl -OL https://github.com/lightvector/KataGo/releases/download/v1.1/g104-b6c96-s97778688-d23397744.zip
unzip g104-b6c96-s97778688-d23397744.zip
cd ../tfjs
# in disposable Python environment
pip install tensorflow==1.13.1
make saved_model/saved_model.pb
pipenv install
pipenv shell
make
cd ..
```

### Build

```sh
source /your/path/emsdk_env.sh
source em_build.sh
```

### Start a web server

```sh
cd web
http-server . # or your favorite one liner server
```

### Open Browser
for auto detection of backend,
```
http://127.0.0.1:8080/?config=gtp_auto.cfg&model=web_model
```
for CPU backend,
```
http://127.0.0.1:8080/?config=gtp_webgl.cfg&model=web_model
```
for WebGL backend (requiring OffscreenCanvas, i.e. Chrome 69 or later), 
```
http://127.0.0.1:8080/?config=gtp_cpu.cfg&model=web_model
```

Enjoy!
-
by ICHIKAWA, Yuji
