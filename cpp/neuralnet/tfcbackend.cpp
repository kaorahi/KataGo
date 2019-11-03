#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"

#include <Model.h>
#include <Tensor.h>

using namespace std;

//---------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

void NeuralNet::globalCleanup() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

//---------------------------------------------------------------------------------------

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  ConfigParser& cfg,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)cfg;
  (void)logger;
  (void)nnXLen;
  (void)nnYLen;
  (void)loadedModel;
  return NULL;
}
void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  (void)computeContext;
  assert(computeContext == NULL);
}

//---------------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;
  string name;
  int numMiscValueChannels;

  LoadedModel(const string& fileName) {
    name = fileName;
    modelDesc.version = 5;
    modelDesc.numInputChannels = 22;
    modelDesc.numInputGlobalChannels = 14;
    modelDesc.numValueChannels = 3;
    modelDesc.numOwnershipChannels = 1;
    modelDesc.numScoreValueChannels = 2;
    numMiscValueChannels = 6;
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  (void)modelFileIdx;
  return new LoadedModel(file);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

//---------------------------------------------------------------------------------------

struct ComputeHandle {
  const LoadedModel* model;
  Model* tfmodel;
  int policySize;
  int nnXLen;
  int nnYLen;

  ComputeHandle(const LoadedModel* loadedModel, int nnX, int nnY) {
    model = loadedModel;
    nnXLen = nnX;
    nnYLen = nnY;
    policySize = NNPos::getPolicySize(nnXLen, nnYLen);
    static Model _tfmodel(model->name);
    tfmodel = &_tfmodel;
  }
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  int nnXLen,
  int nnYLen,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  bool useFP16,
  bool useNHWC
) {
  (void)context;
  (void)logger;
  (void)maxBatchSize;
  (void)requireExactNNLen;
  (void)inputsUseNHWC;
  (void)gpuIdxForThisThread;
  (void)useFP16;
  (void)useNHWC;
  return new ComputeHandle(loadedModel, nnXLen, nnYLen);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

//---------------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer; //Host pointer
  float* userInputGlobalBuffer; //Host pointer
  bool* symmetriesBuffer; //Host pointer

  float* policyResults; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = nnXLen;
    int ySize = nnYLen;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * xSize * ySize;
    singleInputBytes = (size_t)m.numInputChannels * xSize * ySize * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = (size_t)m.numInputGlobalChannels * sizeof(float);
    singlePolicyResultElts = (size_t)(1 + xSize * ySize);
    singlePolicyResultBytes = (size_t)(1 + xSize * ySize) * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = (size_t)m.numValueChannels * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = (size_t)m.numScoreValueChannels * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * xSize * ySize;
    singleOwnershipResultBytes = (size_t)m.numOwnershipChannels * xSize * ySize * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSize * xSize * ySize * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSize * (1 + xSize * ySize) * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSize * xSize * ySize * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[(size_t)m.numInputChannels * maxBatchSize * xSize * ySize];
    userInputGlobalBuffer = new float[(size_t)m.numInputGlobalChannels * maxBatchSize];
    symmetriesBuffer = new bool[NNInputs::NUM_SYMMETRY_BOOLS];

    policyResults = new float[(size_t)maxBatchSize * (1 + xSize * ySize)];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * xSize * ySize * m.numOwnershipChannels];

  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] symmetriesBuffer;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

float* NeuralNet::getBatchEltSpatialInplace(InputBuffers* inputBuffers, int nIdx) {
  assert(nIdx < inputBuffers->maxBatchSize);
  return inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
}

float* NeuralNet::getBatchEltGlobalInplace(InputBuffers* inputBuffers, int nIdx) {
  assert(nIdx < inputBuffers->maxBatchSize);
  return inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* inputBuffers) {
  return inputBuffers->symmetriesBuffer;
}

int NeuralNet::getBatchEltSpatialLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputElts;
}

int NeuralNet::getBatchEltGlobalLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputGlobalElts;
}

//---------------------------------------------------------------------------------------

void copyToTensor(float* source, Tensor* dest, int len);

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* buffers,
  int numBatchEltsFilled,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= buffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  int batchSize = numBatchEltsFilled;
  auto model = gpuHandle->model;
  auto tfmodel = gpuHandle->tfmodel;
  auto nnXLen = gpuHandle->nnXLen;
  auto nnYLen = gpuHandle->nnYLen;
  int version = model->modelDesc.version;

  auto binInputs = new Tensor(*tfmodel, "swa_model/bin_inputs");
  auto globalInputs = new Tensor(*tfmodel, "swa_model/global_inputs");
  auto policyOutput = new Tensor(*tfmodel, "swa_model/policy_output");
  auto valueOutput = new Tensor(*tfmodel, "swa_model/value_output");
  auto miscvaluesOutput = new Tensor(*tfmodel, "swa_model/miscvalues_output");
  // auto scorebeliefOutput = new Tensor(*tfmodel, "swa_model/scorebelief_output");
  // auto bonusbeliefOutput = new Tensor(*tfmodel, "swa_model/bonusbelief_output");
  auto ownershipOutput = new Tensor(*tfmodel, "swa_model/ownership_output");

  copyToTensor(buffers->userInputBuffer, binInputs,
               nnXLen * nnYLen * model->modelDesc.numInputChannels * batchSize);
  copyToTensor(buffers->userInputGlobalBuffer, globalInputs,
               model->modelDesc.numInputGlobalChannels * batchSize);
  tfmodel->run({binInputs, globalInputs},
               {policyOutput, valueOutput, miscvaluesOutput,
                // scorebeliefOutput, bonusbeliefOutput,
                ownershipOutput});

  auto value = valueOutput->get_data<float>();
  auto policy = policyOutput->get_data<float>();
  auto ownership = ownershipOutput->get_data<float>();
  auto miscvalue = miscvaluesOutput->get_data<float>();

  int valueUnit = model->modelDesc.numValueChannels;
  int policyUnit = gpuHandle->policySize * 2;
  int ownershipUnit = nnXLen * nnYLen * model->modelDesc.numOwnershipChannels;
  int miscvalueUnit = model->numMiscValueChannels;

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    int valueOffset = valueUnit * row;
    int policyOffset = policyUnit * row;
    int ownershipOffset = ownershipUnit * row;
    int miscvalueOffset = miscvalueUnit * row;

    output->whiteWinProb = value[0 + valueOffset];
    output->whiteLossProb = value[1 + valueOffset];
    output->whiteNoResultProb = value[2 + valueOffset];

    float* policyProbs = output->policyProbs;
    for (auto i = 0; i < gpuHandle->policySize; i++) {
      policyProbs[i] = policy[2 * i + policyOffset];
    }

    if(output->whiteOwnerMap != NULL) {
      assert(model->modelDesc.numOwnershipChannels == 1);
      for (auto i = 0; i < ownershipUnit; i++) {
        output->whiteOwnerMap[i] = ownership[i + ownershipOffset];
      }
    }

    if(version >= 4) {
      int numScoreValueChannels = model->modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = miscvalue[0 + miscvalueOffset];
      output->whiteScoreMeanSq = miscvalue[1 + miscvalueOffset];
    }
    else if(version >= 3) {
      int numScoreValueChannels = model->modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = miscvalue[0 + miscvalueOffset];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

void copyToTensor(float* source, Tensor* dest, int len) {
  std::vector<float> buf(len);
  for(auto i = 0; i < buf.end() - buf.begin(); i++) {
    buf[i] = source[i];
  }
  dest->set_data(buf);
}

//---------------------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}

//Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateSymmetry(
  int batchSize,
  int numChannels,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const bool* symmetriesBuffer,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer
) {
  (void)batchSize;
  (void)numChannels;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)symmetriesBuffer;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}
