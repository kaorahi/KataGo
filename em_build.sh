# use latest-upstream of emscripten

build_dir=em_build
src_dir=../cpp

if [ ! -d $build_dir ]; then
    mkdir $build_dir
fi
cd $build_dir
emconfigure cmake $src_dir -DBUILD_MCTS=1 -DUSE_BACKEND=TFJS
emmake make
