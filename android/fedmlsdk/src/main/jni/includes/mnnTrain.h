#ifdef FEDML_ANDROID_MNNTRAIN_H
#define FEDML_ANDROID_MNNTRAIN_H

#include <android/bitmap.h>
#include <jni.h>
#include <cstring>
#include <iostream>
#include <iostream>
#include "mnist.h"
#include "cifar10.h"
#include <MNN/expr/Executor.hpp>
#include "SGD.hpp"
#include <MNN/AutoTime.hpp>
#include "Loss.hpp"
#include "LearningRateScheduler.hpp"
#include "Transformer.hpp"
#include "NN.hpp"


#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jstring JNICALL
Java_ai_fedml_edge_nativemnn_NativeMnn_getEpochAndLoss(JNIEnv *env, jclass clazz);

extern "C"
JNIEXPORT jboolean JNICALL
Java_ai_fedml_edge_nativemnn_NativeMnn_stopTraining(JNIEnv *env, jclass);

JNIEXPORT jstring JNICALL
Java_ai_fedml_edge_nativemnn_NativeMnn_train(JNIEnv *env, jclass,
                                             jstring modelCachePath, jstring dataCachePath, jstring dataSet,
                                             jint trainSize, jint testSize, 
                                             jint batchSizeNum, jdouble learningRate, jint epochNum,
                                             jobject listener);
#ifdef __cplusplus
}
#endif
#endif //FEDML_ANDROID_MNNTRAIN_H
