// The reason why ES5 is https://github.com/emscripten-core/emscripten/issues/9190

var GraphModelWrapper = function() {
    this.model = null;
};

GraphModelWrapper.prototype.AUTO = 0;
GraphModelWrapper.prototype.CPU = 1;
GraphModelWrapper.prototype.WEBGL = 2;

GraphModelWrapper.prototype.getBackend = function() {
    switch (tf.getBackend()) {
        case "cpu":
        return this.CPU;
        case "webgl":
        return this.WEBGL;
        default:
        return 0;
    }
}

GraphModelWrapper.prototype.setBackend = function(backend) {
    var be;
    switch (backend) {
        case this.AUTO:
        be = typeof OffscreenCanvas !== 'undefined' ? "webgl" : "cpu";
        break;
        case this.CPU:
        be = "cpu";
        break;
        case this.WEBGL:
        be = "webgl";
        break;
        default:
        return;
    }
    tf.setBackend(be);
};

GraphModelWrapper.prototype.downloadModel = function(charp) {
    return Asyncify.handleSleep((function(wakeUp) {
        const model = UTF8ToString(charp);
        tf.loadGraphModel(model + "/model.json")
            .then((function(model) {
                this.model = model;
                wakeUp(1);
            }).bind(this))
            .catch(function(error) {
                console.error(error);
                wakeUp(0);
            });
    }).bind(this));
};

GraphModelWrapper.prototype.removeModel = function() {

};

GraphModelWrapper.prototype.predict = function(
    batches,
    inputBuffer, inputBufferLength, inputBufferChannels,
    inputGlobalBuffer, inputGlobalBufferChannels,
    values, miscvalues, ownerships, bonusbelieves, scorebelieves, policies) {
    return Asyncify.handleSleep(function(wakeUp) {
        try {
            const bin_inputs = new Float32Array(Module.HEAPF32.buffer, inputBuffer, batches * inputBufferLength * inputBufferChannels);
            const global_inputs = new Float32Array(Module.HEAPF32.buffer, inputGlobalBuffer, batches * inputGlobalBufferChannels);
            this.model.executeAsync({
                "swa_model/bin_inputs": tf.tensor(bin_inputs, [batches, inputBufferLength, inputBufferChannels], 'float32'),
                "swa_model/global_inputs": tf.tensor(global_inputs, [batches, inputGlobalBufferChannels], 'float32')
            }).then(function(results) {
                var i;
                for (i = 0; i < results.length; i++) {
                    const result = results[i];
                    const data = result.dataSync();
                    switch (result.size) {
                        case 3: //value
                        Module.HEAPF32.set(data, values / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                        case 6: // miscvalues
                        Module.HEAPF32.set(data, miscvalues / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                        case 361: // ownership
                        Module.HEAPF32.set(data, ownerships / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                        case 61:  // bonusbelief
                        Module.HEAPF32.set(data, bonusbelieves / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                        case 842: // scorebelief
                        Module.HEAPF32.set(data, scorebelieves / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                        case 724: // policy
                        Module.HEAPF32.set(data, policies / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                    }
                }
                wakeUp(1);
            });
        } catch (e) {
            console.error(e);
            wakeUp(0);
        }
    }.bind(this));
};

GraphModelWrapper.prototype.getModelVersion = function() {
    return 5;
};

if (Module['ENVIRONMENT_IS_PTHREAD']) {
    importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js");
    if (typeof OffscreenCanvas !== 'undefined') {
        self.document = {
            createElement: function() {
                return new OffscreenCanvas(640, 480);
            }
        };
        self.window = self;
        self.screen = {
            width: 640,
            height: 480
        };
        self.HTMLVideoElement = function() {};
        self.HTMLImageElement = function() {};
        self.HTMLCanvasElement = OffscreenCanvas;
    } else {
        console.error("no offscreen canvas");
    }
} else {
    function enableInput(status) {
        var command = document.getElementById("input").command
        switch (status) {
            case 1:
            command.removeAttribute("disabled");
            command.setAttribute("placeholder", "GTP command");
            command.focus();
            break;
            case -1:
            command.setAttribute("placeholder", "Engine failed loading a weight");
        }
    }

    var Input = function() {
        this.buffer = "";
    
        document.getElementById("input").addEventListener("submit", (function(event) {
            event.preventDefault();
            this.buffer += event.currentTarget.command.value + "\n";
            document.getElementById("log").value += event.currentTarget.command.value + "\n";
            event.currentTarget.command.value = "";
            if (this.resolve) {
                this.resolve();
            }
        }).bind(this), false);
    };
    
    Input.prototype.callback = function() {
        if (!this.buffer) {
            return null;
        }
        const c = this.buffer[0];
        this.buffer = this.buffer.substr(1);
        return c.charCodeAt(0);
    };
    
    Input.prototype.wait = function() {
        return new Promise((function(res, rej) {
            this.resolve = res;
            this.reject = rej;
        }).bind(this));
    };
    
    
    var Output = function() {
        this.buffer = "";
        this.crFlag = false;
    };
    Output.prototype.callback = function(char) {
        if (char === 0 || char === 0x0a) {
            if (this.buffer.length < 1000) {
                var output = document.getElementById("output")
                output.value += this.buffer + "\n";
                document.getElementById("log").value += this.buffer + "\n";
                output.dispatchEvent(new CustomEvent("message"));
            }
            this.buffer = "";
            this.crFlag = false;
            return;
        }
        if (char === 0x0d) {
            this.crFlag = true;
            return;
        } 
        if (this.crFlag) {
            this.crFlag = false;
            this.buffer = "";
        }
        this.buffer += String.fromCharCode(char);
    };
}

if (!("preRun" in Module)) {
    Module["preRun"] = [];
}
Module["preRun"].push(function() {
    console.log("preRun");
    var params = new URL(location).searchParams;
    var cfgFile = params.get("config") || "gtp_auto.cfg";
    FS.createPreloadedFile(
        FS.cwd(),
        cfgFile,
        cfgFile,
        true, // 読み込み許可
        false // 書き込み許可
    );
    if (!("arguments" in Module)) {
        Module["arguments"] = [];
    }
    Module["arguments"].push(params.get("subcommand") || "gtp");
    Module["arguments"].push("-model");
    Module["arguments"].push(params.get("model") || "web_model");
    Module["arguments"].push("-config");
    Module["arguments"].push(cfgFile);

    var input = new Input();
    var output = new Output();
    FS.init(input.callback.bind(input), output.callback.bind(output), null);
    Module["input"] = input;
    Module["output"] = output;
});