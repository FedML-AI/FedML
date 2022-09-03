#include <jni.h>
#include "jniAssist.h"
#include "JavaArrayList.h"
#include "JavaBundle.h"
#include "JVMContainer.h"

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
	JNIEnv* env = nullptr;
	jint result = -1;

	if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK){
		result = JNI_VERSION_1_6;
	} else if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_4) == JNI_OK) {
		result = JNI_VERSION_1_4;
	} else {
	    return	JNI_ERR;
	}
    
    JVMContainer::InitVM (vm,result) ;
    JavaArrayList::ensureArrayListClassNotNull();
    JavaBundle::ensureBundleClassNotNull();
    LOGI("JNI_OnLoad SUCCESS!");
	return result;
}