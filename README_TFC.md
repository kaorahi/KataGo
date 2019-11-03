# TensorFlow C API backend

## Install

Example for Debian GNU/Linux (buster)

### Convert a weight for cppflow

```sh
sudo apt-get install python3-pip
pip3 install tensorflowjs==1.2.3

mkdir -p tfc/models
cd tfc/models
curl -OL https://github.com/lightvector/KataGo/releases/download/v1.1/g104-b6c96-s97778688-d23397744.zip
unzip g104-b6c96-s97778688-d23397744.zip
cd ../..

cd tfjs
python3 save_graph.py -freeze -name-scope swa_model -model-variables-prefix ../tfc/models/g104-b6c96-s97778688-d23397744/saved_model/variables/variables -model-config-json ../tfc/models/g104-b6c96-s97778688-d23397744/model.config.json
mv frozen_graph.pb ../tfc/models/frozen_g104-b6c96.pb
cd ..
```

### Download libtensorflow

```sh
mkdir -p tfc/libtensorflow
curl -s https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz | tar xvfz - -C tfc/libtensorflow
```

### Download cppflow

```sh
mkdir -p tfc/cppflow
cd tfc/cppflow
curl -OL https://github.com/serizba/cppflow/archive/master.zip
unzip master.zip
mv cppflow-master/ ../../cpp/cppflow
cd ../..
```

(The version at [2019-11-03] is `https://github.com/serizba/cppflow/archive/4c84bb1622f5f5a6b400e8839c05cb35b7842a8a.zip`.)

### Compile & Run

```sh
cd cpp
cmake . -DBUILD_MCTS=1 -DUSE_BACKEND=TFC
make
echo "time_settings 0 1 1\ngenmove b\ngenmove w\nshowboard" | ./katago gtp -model ../tfc/models/frozen_g104-b6c96.pb -config configs/gtp_example.cfg
```
