#include "includes/jniAssist.h"
#include <cstdlib>

/**
 * Equivalent to ScopedLocalRef, but for C_JNIEnv instead. (And slightly more powerful.)
 */
template<typename T>
class scoped_local_ref {
public:
    explicit scoped_local_ref(C_JNIEnv *env, T localRef = NULL)
            : mEnv(env), mLocalRef(localRef) {
    }

    // Disallow copy and assignment.
    scoped_local_ref(const scoped_local_ref &) = delete;

    void operator=(const scoped_local_ref &) = delete;

    ~scoped_local_ref() {
        reset();
    }

    void reset(T localRef = NULL) {
        if (mLocalRef != NULL) {
            (*mEnv)->DeleteLocalRef(reinterpret_cast<JNIEnv *>(mEnv), mLocalRef);
            mLocalRef = localRef;
        }
    }

    T get() const {
        return mLocalRef;
    }

private:
    C_JNIEnv *mEnv;
    T mLocalRef;
};

static jclass findClass(C_JNIEnv *env, const char *className) {
    auto *e = reinterpret_cast<JNIEnv *>(env);
    return (*env)->FindClass(e, className);
}

extern "C" int jniRegisterNativeMethods(C_JNIEnv *env, const char *className,
                                        const JNINativeMethod *gMethods, int numMethods) {
    auto *e = reinterpret_cast<JNIEnv *>(env);

    LOGD("Registering %s natives", className);

    scoped_local_ref<jclass> c(env, findClass(env, className));
    if (c.get() == NULL) {
        LOGD("Native registration unable to find class '%s', aborting", className);
        abort();
    }

    if ((*env)->RegisterNatives(e, c.get(), gMethods, numMethods) < 0) {
        LOGD("RegisterNatives failed for '%s', aborting", className);
        abort();
    }
    return 0;
}

extern "C"
jmethodID getMethodIdByNameAndSig(JNIEnv *env, jobject thiz, const char *name, const char *sig) {
    if (env == nullptr || thiz == nullptr) {
        return nullptr;
    }
    jclass subscriberClass = env->GetObjectClass(thiz);
    jmethodID methodId = env->GetMethodID(subscriberClass, name, sig);
    if (methodId == nullptr) {
        return nullptr;
    }
    return methodId;
}