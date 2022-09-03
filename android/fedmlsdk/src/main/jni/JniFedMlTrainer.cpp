//
// Created by flex on 2022/6/26.
//
#ifdef FEDML_ANDROID_JNIFEDMLTRAINER_CPP
#define FEDML_ANDROID_JNIFEDMLTRAINER_CPP

#include "JniFedMlTrainer.h"
#include "jniAssist.h"
#include "FedMLClientManagerSA.h"

#ifdef __cplusplus
extern "C" {
#endif

std::map<jlong, jobject> globalCallbackMap;


JNIEXPORT jlong JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_create
        (JNIEnv *, jobject) {
    LOGD("NativeFedMLTrainer.create");
    return reinterpret_cast<jlong >(new FedMLClientManagerSA());
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    release
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_release
        (JNIEnv *env, jobject, jlong ptr) {
    LOGD("NativeFedMLTrainer<%lx>.release", ptr);
    jobject globalCallback = globalCallbackMap[ptr];
    if (globalCallback != nullptr) {
        globalCallbackMap.erase(ptr);
        env->DeleteGlobalRef(globalCallback);
    }
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    delete pFedMLClientManager;
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    init
 * Signature: (ILjava/lang/String;Ljava/lang/String;Ljava/lang/String;IIIDIIIILai/fedml/edge/nativemnn/TrainingCallback;)V
 */
JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_init
        (JNIEnv *env, jobject , jlong ptr, jstring modelCachePath, jstring dataCachePath,
         jstring dataSet,
         jint trainSize, jint testSize, jint batchSizeNum, jdouble learningRate, jint epochNum,
         jint q_bits, jint p, jint client_num, jobject trainingCallback) {
    // model and dataset path
    LOGD("===================JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_init=========");
    const char *modelPath = env->GetStringUTFChars(modelCachePath, nullptr);
    const char *dataPath = env->GetStringUTFChars(dataCachePath, nullptr);
    const char *datasetType = env->GetStringUTFChars(dataSet, nullptr);
    LOGD("NativeFedMLTrainer<%lx>.init(%s, %s, %s, %d, %d, %d, %f, %d, %d, %d, %d)", ptr,
         modelPath, dataPath, datasetType, trainSize, testSize,
         batchSizeNum, learningRate, epochNum, q_bits, p, client_num);
    // callback function
    jobject globalCallback = env->NewGlobalRef(trainingCallback);
    globalCallbackMap[ptr] = globalCallback;
    jmethodID onProgressMethodID = getMethodIdByNameAndSig(env, trainingCallback, "onProgress",
                                                           "(F)V");
    jmethodID onLossMethodID = getMethodIdByNameAndSig(env, trainingCallback, "onLoss", "(IF)V");
    jmethodID onAccuracyMethodID = getMethodIdByNameAndSig(env, trainingCallback, "onAccuracy",
                                                           "(IF)V");
    LOGD("NativeFedMLTrainer<%lx>.init onProgressMid=%p,onLossMid=%p,onAccuracyMid=%p", ptr,
         onProgressMethodID, onLossMethodID, onAccuracyMethodID);
    auto onProgressCallback = [ptr, env, onProgressMethodID](float progress) {
        jobject callback = globalCallbackMap[ptr];
        LOGD("NativeFedMLTrainer<%lx> <%p>.onProgressCallback(%f) env=%p onProgressMid=%p", ptr,
             callback, progress, env, onProgressMethodID);
        env->CallVoidMethod(callback, onProgressMethodID, (jfloat) progress);
    };
    auto onLossCallback = [ptr, env, onLossMethodID](int epoch, float loss) {
        jobject callback = globalCallbackMap[ptr];
        LOGD("NativeFedMLTrainer<%lx> <%p>.onLossCallback(%d, %f) env=%p onLossMid=%p", ptr,
             callback, epoch, loss, env, onLossMethodID);
        env->CallVoidMethod(callback, onLossMethodID, (jint) epoch, (jfloat) loss);
    };
    auto onAccuracyCallback = [ptr, env, onAccuracyMethodID](int epoch, float acc) {
        jobject callback = globalCallbackMap[ptr];
        LOGD("NativeFedMLTrainer<%lx> <%p>.onLossCallback(%d, %f) env=%p onLossMid=%p", ptr,
             callback, epoch, acc, env, onAccuracyMethodID);
        env->CallVoidMethod(callback, onAccuracyMethodID, (jint) epoch, (jfloat) acc);
    };
    // FedMLClientManagerSA object ptr
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    pFedMLClientManager->init(modelPath, dataPath, datasetType, (int) trainSize, (int) testSize,
                              (int) batchSizeNum, (double) learningRate, (int) epochNum,
                              (int) q_bits, (int) p, (int) client_num,
                              onProgressCallback, onLossCallback, onAccuracyCallback);
    LOGD("NativeFedMLTrainer<%lx>.initialed", ptr);
    onProgressCallback(0.01);
    env->ReleaseStringUTFChars(modelCachePath, modelPath);
    env->ReleaseStringUTFChars(dataCachePath, dataPath);
    env->ReleaseStringUTFChars(dataSet, datasetType);
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    getLocalEncodedMask
 * Signature: ()[[F
 */
JNIEXPORT jobjectArray JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_getLocalEncodedMask
        (JNIEnv *env, jobject, jlong ptr) {
    LOGD("NativeFedMLTrainer<%lx>.getLocalEncodedMask", ptr);
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    std::vector<std::vector<float>> mask = pFedMLClientManager->get_local_encoded_mask();
    LOGD("NativeFedMLTrainer<%lx>.getLocalEncodedMask encoded_mask[%ld][%ld]", ptr, mask.size(),
         mask[0].size());
    jclass floatArrClass = env->FindClass("[F");
    jobjectArray maskMatrix = env->NewObjectArray(mask.size(), floatArrClass, nullptr);
    jsize maskSize = mask.size();
    for (jsize i = 0; i < maskSize; i++) {
        std::vector<float> rowMask = mask[i];
        jsize arrSize = rowMask.size();
        jfloatArray maskArr = env->NewFloatArray(arrSize);
        env->SetFloatArrayRegion(maskArr, 0, arrSize, rowMask.data());
        env->SetObjectArrayElement(maskMatrix, i, maskArr);
        env->DeleteLocalRef(maskArr);
    }
    return maskMatrix;
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    saveMaskFromPairedClients
 * Signature: (I[F)V
 */
JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_saveMaskFromPairedClients
        (JNIEnv *env, jobject, jlong ptr, jint clientIndex, jfloatArray localEncodeMask) {
    LOGD("NativeFedMLTrainer<%lx>.saveMaskFromPairedClients", ptr);
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    // set value of localEncodeMask to float array
    int len = env->GetArrayLength(localEncodeMask);
    float maskValues[len];
    env->GetFloatArrayRegion(localEncodeMask, 0, len, (jfloat *) maskValues);
    // use value of maskValues to initial encodeMaskVec
    std::vector<float> encodeMaskVec(maskValues, maskValues + len);
    pFedMLClientManager->save_mask_from_paired_clients((int) clientIndex, encodeMaskVec);
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    getClientIdsThatHaveSentMask
 * Signature: ()[I
 */
JNIEXPORT jintArray JNICALL
Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_getClientIdsThatHaveSentMask
        (JNIEnv *env, jobject, jlong ptr) {
    LOGD("NativeFedMLTrainer<%lx>.getClientIdsThatHaveSentMask", ptr);
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    std::vector<int> ids = pFedMLClientManager->get_client_IDs_that_have_sent_mask();
    jintArray idArr = env->NewIntArray(ids.size());
    env->SetIntArrayRegion(idArr, 0, ids.size(), ids.data());
    return idArr;
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    train
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_train
        (JNIEnv *env, jobject, jlong ptr) {
    LOGD("NativeFedMLTrainer<%lx>.train", ptr);
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    std::string result = pFedMLClientManager->train();
    return env->NewStringUTF(result.data());
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    generateMaskedModel
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_generateMaskedModel
        (JNIEnv *, jobject, jlong ptr) {
    LOGD("NativeFedMLTrainer<%lx>.generateMaskedModel", ptr);
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    pFedMLClientManager->generate_masked_model();
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    getAggregatedEncodedMask
 * Signature: ([I)[F
 */
JNIEXPORT jfloatArray JNICALL
Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_getAggregatedEncodedMask
        (JNIEnv *env, jobject, jlong ptr, jintArray survivingListFromServer) {
    LOGD("NativeFedMLTrainer<%lx>.getAggregatedEncodedMask", ptr);
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    int len = env->GetArrayLength(survivingListFromServer);
    int survivingList[len];
    env->GetIntArrayRegion(survivingListFromServer, 0, len, (jint *) survivingList);
    std::vector<int> survivingVec(survivingList, survivingList + len);
    std::vector<float> maskVec = pFedMLClientManager->get_aggregated_encoded_mask(survivingVec);
    jfloatArray maskArr = env->NewFloatArray(maskVec.size());
    env->SetFloatArrayRegion(maskArr, 0, maskVec.size(), maskVec.data());
    return maskArr;
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    getEpochAndLoss
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_getEpochAndLoss
        (JNIEnv *env, jobject, jlong ptr) {
    LOGD("NativeFedMLTrainer<%lx>.getEpochAndLoss", ptr);
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    std::string result = pFedMLClientManager->getEpochAndLoss();
    return env->NewStringUTF(result.data());
}

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    stopTraining
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_stopTraining
        (JNIEnv *, jobject, jlong ptr) {
    LOGD("NativeFedMLTrainer<%lx>.stopTraining", ptr);
    auto *pFedMLClientManager = reinterpret_cast<FedMLClientManagerSA *>(ptr);
    return pFedMLClientManager->stopTraining();
}

#ifdef __cplusplus
}
#endif
#endif //FEDML_ANDROID_JNIFEDMLTRAINER_CPP