function softmax(input, temperature = 1.0) {
    const alpha = Math.max.apply(null, input);
    const powers = input.map(e => Math.exp((e - alpha) / temperature));
    const denom = powers.reduce((sum, e) => sum + e);
    return powers.map(e => e / denom);
}

async function test() {
    const batches = 1;
    const inputBufferLength = 19 * 19;
    const inputBufferChannels = 22;
    const inputGlobalBufferChannels = 14;
    const bin_inputs = new Float32Array(batches * inputBufferLength * inputBufferChannels);
    const global_inputs = new Float32Array(batches * inputGlobalBufferChannels);
    for (let y = 0; y < 19; y++) {
        for (let x = 0; x < 19; x++) {
            bin_inputs[inputBufferChannels * (19 * y + x)] = 1.0;
        }
    }
    global_inputs[5] = -0.5; // コミ 15目=1.0
    global_inputs[6] = 1; // positional ko
    global_inputs[7] = 0.5; // positional ko
    global_inputs[8] = 1; // multiStoneSuicideLegal
    global_inputs[13] = -0.5; // ?
    try {
        tf.setBackend("cpu");
        const model = await tf.loadGraphModel("./web_model/model.json");
        const results = await model.executeAsync({
            "swa_model/bin_inputs": tf.tensor(bin_inputs, [batches, inputBufferLength, inputBufferChannels], 'float32'),
            "swa_model/global_inputs": tf.tensor(global_inputs, [batches, inputGlobalBufferChannels], 'float32')
        });
        results[0].print();
        const value = results[0].as1D().softmax();
        value.print();
        const policy = results[5].slice([0, 0, 0], [1, -1, 1]).as1D();
        policy.print();
        /*
        const d = policy.softmax();
        d.print();
        const data = d.dataSync();
        */
        //const data = softmax(policy.dataSync());
        const data = policy.dataSync();
        for (let y = 0; y < 19; y++) {
            let str = (" " + y).slice(-2) + " ";
            for (let x = 0; x < 19; x++) {
                str += (data[19 * y + x] * 100).toFixed(0) + " ";
            }
            console.log(str);
        }
        console.log((data[data.length - 1] * 100).toFixed(0));
    } catch (e) {
        console.log(e);
    }
}

test();