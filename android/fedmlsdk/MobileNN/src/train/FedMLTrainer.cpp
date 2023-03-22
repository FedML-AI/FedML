#include "FedMLTrainer.h"

FedMLTrainer::FedMLTrainer() {
#ifdef USE_MNN_BACKEND
    printf("using MNN as backend\n");
    m_trainer = new FedMLMNNTrainer();
#endif
#ifdef USE_TORCH_BACKEND
    printf("using torch as backend\n");
    m_trainer = new FedMLTorchTrainer();
#endif
}