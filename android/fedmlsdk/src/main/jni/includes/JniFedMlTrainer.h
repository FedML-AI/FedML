//
// Created by Flex on 2022/6/26.
//
#ifdef FEDML_ANDROID_JNIFEDMLTRAINER_H
#define FEDML_ANDROID_JNIFEDMLTRAINER_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    create
 * Signature: ()I
 */
JNIEXPORT jlong JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_create
        (JNIEnv *, jobject);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    release
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_release
        (JNIEnv *, jobject, jlong);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    init
 * Signature: (ILjava/lang/String;Ljava/lang/String;Ljava/lang/String;IIIDIIIILai/fedml/edge/nativemnn/TrainingCallback;)V
 */
JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_init
        (JNIEnv *, jobject, jlong, jstring, jstring, jstring, jint, jint, jint, jdouble, jint, jint,
         jint, jint, jobject);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    getLocalEncodedMask
 * Signature: ()[[F
 */
JNIEXPORT jobjectArray JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_getLocalEncodedMask
        (JNIEnv *, jobject, jlong);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    saveMaskFromPairedClients
 * Signature: (I[F)V
 */
JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_saveMaskFromPairedClients
        (JNIEnv *, jobject, jlong, jint, jfloatArray);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    getClientIdsThatHaveSentMask
 * Signature: ()[I
 */
JNIEXPORT jintArray JNICALL
Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_getClientIdsThatHaveSentMask
        (JNIEnv *, jobject, jlong);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    train
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_train
        (JNIEnv *, jobject, jlong);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    generateMaskedModel
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_generateMaskedModel
        (JNIEnv *, jobject, jlong);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    getAggregatedEncodedMask
 * Signature: ([I)[F
 */
JNIEXPORT jfloatArray JNICALL
Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_getAggregatedEncodedMask
        (JNIEnv *, jobject, jlong, jintArray);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    getEpochAndLoss
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_getEpochAndLoss
        (JNIEnv *, jobject, jlong);

/*
 * Class:     ai_fedml_edge_nativemnn_NativeFedMLTrainer
 * Method:    stopTraining
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_ai_fedml_edge_nativemnn_NativeFedMLTrainer_stopTraining
        (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif //FEDML_ANDROID_JNIFEDMLTRAINER_H