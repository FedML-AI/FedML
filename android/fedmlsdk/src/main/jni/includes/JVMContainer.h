#ifndef __ANDROID_JVM_CONTAINER_H__
#define __ANDROID_JVM_CONTAINER_H__

#include "jni.h"

class JVMContainer
{
private:
	static JavaVM *global_JVM;
	static jint sJvmVersion;

public:
	static void InitVM(JavaVM *vm,jint jniVersion);
	static JavaVM* GetJVM();
	static void GetEnvironment(JNIEnv **env);
	static jint GetJvmVersion();
};

jmethodID GetMethodID(JNIEnv *env, jclass clazz, const char *name, const char *signature);
jmethodID GetStaticMethodID(JNIEnv *env, jclass clazz, const char *name, const char *signature);

#endif
