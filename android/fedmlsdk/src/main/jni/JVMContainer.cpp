#include "JVMContainer.h"

JavaVM* JVMContainer::global_JVM = nullptr;
jint    JVMContainer::sJvmVersion = 0;

void JVMContainer::InitVM(JavaVM *vm, jint jvmVersion)
{
	if (!global_JVM)
	{
		global_JVM = vm;
		sJvmVersion = jvmVersion;
	}
}

JavaVM* JVMContainer::GetJVM()
{
	return global_JVM;
}

void JVMContainer::GetEnvironment(JNIEnv **env)
{
	if (global_JVM)
	{
		global_JVM->AttachCurrentThread(env, 0);
	}
}

jint JVMContainer::GetJvmVersion()
{
	return sJvmVersion;
}

jmethodID GetMethodID(JNIEnv *env, jclass clazz, const char *name, const char *signature){
	jmethodID mid = 0;
	if(env && clazz){
		mid = env->GetMethodID(clazz, name, signature);
	}
	
	if (env && env->ExceptionCheck()) {
		env->ExceptionDescribe();
		env->ExceptionClear();
	}
	return mid;
}

jmethodID GetStaticMethodID(JNIEnv *env, jclass clazz, const char *name, const char *signature){
	jmethodID mid = 0;
	if(env && clazz){
		mid = env->GetStaticMethodID(clazz, name, signature);
	}
	
	if (env && env->ExceptionCheck()) {
		env->ExceptionDescribe();
		env->ExceptionClear();
	}
	return mid;
}
