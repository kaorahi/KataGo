

mergeInto(LibraryManager.library, {
    notifyStatus: function(status) {
        enableInput(status); // cannot pass directly since enableInput would be defined dynamically.
    },
    $stdio_support__postset: 'stdio_support();',
    $stdio_support: function() {
        if (!Module['ENVIRONMENT_IS_PTHREAD']) {
           _waitForStdin = function() {
                Asyncify.handleSleep(function(wakeUp) {
                    Module["input"].wait().then(function() {
                        wakeUp();
                    });
                });
            };
        }
    },
    waitForStdin: function() {
        console.log("waitForStdin. should not reach");
    },
    waitForStdin__deps: ['$stdio_support'],

    $method_support__postset: 'method_support();',
    $method_support: function() {
        const inst = new GraphModelWrapper();
        _getBackend = inst.getBackend.bind(inst);
        _setBackend = inst.setBackend.bind(inst);
        _downloadModel = inst.downloadModel.bind(inst);
        _removeModel = inst.removeModel.bind(inst);
        _predict = inst.predict.bind(inst);
        _jsGetModelVersion = inst.getModelVersion.bind(inst);
    },
    getBackend: function() {
        console.log("getBackend. should not reach");
    },
    getBackend__deps: ['$method_support'],
    setBackend: function() {
        console.log("setBackend. should not reach");
    },
    setBackend__deps: ['$method_support'],
    downloadModel: function() {
        console.log("downloadModel. should not reach");
    },
    downloadModel__deps: ['$method_support'],
    removeModel: function() {
        console.log("removeModel. should not reach");
    },
    removeModel__deps: ['$method_support'],
    predict: function() {
        console.log("predict. should not reach");
    },
    predict__deps: ['$method_support'],
    jsGetModelVersion: function() {
        console.log("jsGetModelVersion. should not reach");
    },
    jsGetModelVersion__deps: ['$method_support'],
});
