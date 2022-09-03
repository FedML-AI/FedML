#include "includes/JavaBundle.h"
#include "includes/JVMContainer.h"
#include <cassert>

static jclass Bundle_Class = nullptr;
jmethodID Bundle_constructorFunc = nullptr;
jmethodID Bundle_getIntFunc = nullptr;
jmethodID Bundle_putIntFunc = nullptr;
jmethodID Bundle_getDoubleFunc = nullptr;
jmethodID Bundle_putDoubleFunc = nullptr;
jmethodID Bundle_getJStringFunc = nullptr;
jmethodID Bundle_putStringFunc = nullptr;
jmethodID Bundle_getJStringArrayFunc = nullptr;
jmethodID Bundle_putStringArrayFunc = nullptr;
jmethodID Bundle_getByteArrayFunc = nullptr;
jmethodID Bundle_putByteArrayFunc = nullptr;
jmethodID Bundle_putCharArrayFunc = nullptr;
jmethodID Bundle_getIntArrayFunc = nullptr;
jmethodID Bundle_putIntArrayFunc = nullptr;
jmethodID Bundle_clearFunc = nullptr;
jmethodID Bundle_putLongFunc = nullptr;
jmethodID Bundle_getLongFunc = nullptr;
jmethodID Bundle_putBundleFunc = nullptr;
jmethodID Bundle_getBundleFunc = nullptr;
jmethodID Bundle_putParcelableArrayFunc = nullptr;
jmethodID Bundle_getParcelableArrayFunc = nullptr;
jmethodID Bundle_putParcelableArrayListFunc = nullptr;
jmethodID Bundle_getParcelableArrayListFunc = nullptr;
jmethodID Bundle_putFloatFunc = nullptr;
jmethodID Bundle_getFloatFunc = nullptr;
jmethodID Bundle_containsKeyFunc = nullptr;
jmethodID Bundle_getBooleanFunc = nullptr;
jmethodID Bundle_putBooleanFunc = nullptr;
jmethodID Bundle_putBooleanArrayFunc = nullptr;
jmethodID Bundle_getBooleanArrayFunc = nullptr;
jmethodID Bundle_getByteFunc = nullptr;
jmethodID Bundle_putByteFunc = nullptr;
jmethodID Bundle_getCharFunc = nullptr;
jmethodID Bundle_putCharFunc = nullptr;
jmethodID Bundle_getShortFunc = nullptr;
jmethodID Bundle_putShortFunc = nullptr;
jmethodID Bundle_getShortArrayFunc = nullptr;
jmethodID Bundle_putShortArrayFunc = nullptr;
jmethodID Bundle_removeFunc = nullptr;
jmethodID Bundle_getJStringArrayListFunc = nullptr;
jmethodID Bundle_putStringArrayListFunc = nullptr;
jmethodID Bundle_putFloatArrayFunc = nullptr;
jmethodID Bundle_getFloatArrayFunc = nullptr;
jmethodID Bundle_putDoubleArrayFunc = nullptr;
jmethodID Bundle_getDoubleArrayFunc = nullptr;
jmethodID Bundle_putLongArrayFunc = nullptr;
jmethodID Bundle_getLongArrayFunc = nullptr;
jmethodID Bundle_getCharArrayFunc = nullptr;


#define JAVA_BUNDLE_CLASS_NAME "android/os/Bundle"

JavaBundle::JavaBundle(JNIEnv *jniEnv) {
    mJNIEnv = jniEnv;
    assert(mJNIEnv);

    JavaBundle::ensureBundleClassNotNull();
    if (Bundle_constructorFunc == nullptr) {
        Bundle_constructorFunc = mJNIEnv->GetMethodID(Bundle_Class, "<init>", "()V");
    }
    mJavaBundleObj = mJNIEnv->NewObject(Bundle_Class, Bundle_constructorFunc);
}

JavaBundle::JavaBundle(JNIEnv *jniEnv, jobject javaBundleObject) {
    mJNIEnv = jniEnv;
    assert(mJNIEnv);

    if (javaBundleObject == nullptr) {
        mJNIEnv->ThrowNew(mJNIEnv->FindClass("java/lang/Exception"),
                          "JavaBundle::JavaBundle---javaBundleObject is nullptr");
    }
    JavaBundle::ensureBundleClassNotNull();
    mJavaBundleObj = javaBundleObject;
}

JavaBundle::~JavaBundle() {
    mJNIEnv->DeleteLocalRef(mJavaBundleObj);
    mJNIEnv = nullptr;
}

jint JavaBundle::getInt(jstring key) {
    if (Bundle_getIntFunc == nullptr) {
        Bundle_getIntFunc = mJNIEnv->GetMethodID(Bundle_Class, "getInt", "(Ljava/lang/String;)I");
    }
    return mJNIEnv->CallIntMethod(mJavaBundleObj, Bundle_getIntFunc, key);
}

void JavaBundle::putInt(jstring key, jint value) {
    if (Bundle_putIntFunc == nullptr) {
        Bundle_putIntFunc = mJNIEnv->GetMethodID(Bundle_Class, "putInt", "(Ljava/lang/String;I)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putIntFunc, key, value);
}

jintArray JavaBundle::getIntArray(jstring key) {
    if (Bundle_getIntArrayFunc == nullptr) {
        Bundle_getIntArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getIntArray",
                                                      "(Ljava/lang/String;)[I");
    }
    return (jintArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getIntArrayFunc, key);
}

void JavaBundle::putIntArray(jstring key, jintArray value) {
    if (Bundle_putIntArrayFunc == nullptr) {
        Bundle_putIntArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putIntArray",
                                                      "(Ljava/lang/String;[I)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putIntArrayFunc, key, value);
}

jboolean JavaBundle::getBoolean(jstring key) {
    if (Bundle_getBooleanFunc == nullptr) {
        Bundle_getBooleanFunc = mJNIEnv->GetMethodID(Bundle_Class, "getBoolean",
                                                     "(Ljava/lang/String;)Z");
    }
    return mJNIEnv->CallBooleanMethod(mJavaBundleObj, Bundle_getBooleanFunc, key);
}

void JavaBundle::putBoolean(jstring key, jboolean value) {
    if (Bundle_putBooleanFunc == nullptr) {
        Bundle_putBooleanFunc = mJNIEnv->GetMethodID(Bundle_Class, "putBoolean",
                                                     "(Ljava/lang/String;Z)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putBooleanFunc, key, value);
}

jbooleanArray JavaBundle::getBooleanArray(jstring key) {
    if (Bundle_getBooleanArrayFunc == nullptr) {
        Bundle_getBooleanArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getBooleanArray",
                                                          "(Ljava/lang/String;)[Z");
    }
    return (jbooleanArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getBooleanArrayFunc,
                                                     key);
}

void JavaBundle::putBooleanArray(jstring key, jbooleanArray value) {
    if (Bundle_putBooleanArrayFunc == nullptr) {
        Bundle_putBooleanArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putBooleanArray",
                                                          "(Ljava/lang/String;[Z)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putBooleanArrayFunc, key, value);
}

jshort JavaBundle::getShort(jstring key) {
    if (Bundle_getShortFunc == nullptr) {
        Bundle_getShortFunc = mJNIEnv->GetMethodID(Bundle_Class, "getShort",
                                                   "(Ljava/lang/String;)S");
    }
    return mJNIEnv->CallShortMethod(mJavaBundleObj, Bundle_getShortFunc, key);
}

void JavaBundle::putShort(jstring key, jshort value) {
    if (Bundle_putShortFunc == nullptr) {
        Bundle_putShortFunc = mJNIEnv->GetMethodID(Bundle_Class, "putShort",
                                                   "(Ljava/lang/String;S)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putShortFunc, key, value);
}

jshortArray JavaBundle::getShortArray(jstring key) {
    if (Bundle_getShortArrayFunc == nullptr) {
        Bundle_getShortArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getShortArray",
                                                        "(Ljava/lang/String;)[S");
    }
    return (jshortArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getShortArrayFunc, key);
}

void JavaBundle::putShortArray(jstring key, jshortArray value) {
    if (Bundle_putShortArrayFunc == nullptr) {
        Bundle_putShortArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putShortArray",
                                                        "(Ljava/lang/String;[S)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putShortArrayFunc, key, value);
}

jdouble JavaBundle::getDouble(jstring key) {
    if (Bundle_getDoubleFunc == nullptr) {
        Bundle_getDoubleFunc = mJNIEnv->GetMethodID(Bundle_Class, "getDouble",
                                                    "(Ljava/lang/String;)D");
    }
    return mJNIEnv->CallDoubleMethod(mJavaBundleObj, Bundle_getDoubleFunc, key);
}

void JavaBundle::putDouble(jstring key, jdouble value) {
    if (Bundle_putDoubleFunc == nullptr) {
        Bundle_putDoubleFunc = mJNIEnv->GetMethodID(Bundle_Class, "putDouble",
                                                    "(Ljava/lang/String;D)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putDoubleFunc, key, value);
}

jdoubleArray JavaBundle::getDoubleArray(jstring key) {
    if (Bundle_getDoubleArrayFunc == nullptr) {
        Bundle_getDoubleArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getDoubleArray",
                                                         "(Ljava/lang/String;)[D");
    }
    return (jdoubleArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getDoubleArrayFunc, key);
}

void JavaBundle::putDoubleArray(jstring key, jdoubleArray value) {
    if (Bundle_putDoubleArrayFunc == nullptr) {
        Bundle_putDoubleArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putDoubleArray",
                                                         "(Ljava/lang/String;[D)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putDoubleArrayFunc, key, value);
}

jstring JavaBundle::getJString(jstring key) {
    if (Bundle_getJStringFunc == nullptr) {
        Bundle_getJStringFunc = mJNIEnv->GetMethodID(Bundle_Class, "getString",
                                                     "(Ljava/lang/String;)Ljava/lang/String;");
    }
    return (jstring) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getJStringFunc, key);
}

void JavaBundle::putJString(jstring key, jstring value) {
    if (Bundle_putStringFunc == nullptr) {
        Bundle_putStringFunc = mJNIEnv->GetMethodID(Bundle_Class, "putString",
                                                    "(Ljava/lang/String;Ljava/lang/String;)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putStringFunc, key, value);
}

jobjectArray JavaBundle::getStringArray(jstring key) {
    if (Bundle_getJStringArrayFunc == nullptr) {
        Bundle_getJStringArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getStringArray",
                                                          "(Ljava/lang/String;)[Ljava/lang/String;");
    }
    return (jobjectArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getJStringArrayFunc,
                                                    key);
}

void JavaBundle::putStringArray(jstring key, jobjectArray value) {
    if (Bundle_putStringArrayFunc == nullptr) {
        Bundle_putStringArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putStringArray",
                                                         "(Ljava/lang/String;[Ljava/lang/String;)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putStringArrayFunc, key, value);
}

jobject JavaBundle::getStringArrayList(jstring key) {
    if (Bundle_getJStringArrayListFunc == nullptr) {
        Bundle_getJStringArrayListFunc = mJNIEnv->GetMethodID(Bundle_Class, "getStringArrayList",
                                                              "(Ljava/lang/String;)Ljava/util/ArrayList;");
    }
    return mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getJStringArrayListFunc, key);
}

void JavaBundle::putStringArrayList(jstring key, jobject value) {
    if (Bundle_putStringArrayListFunc == nullptr) {
        Bundle_putStringArrayListFunc = mJNIEnv->GetMethodID(Bundle_Class, "putStringArrayList",
                                                             "(Ljava/lang/String;Ljava/util/ArrayList;)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putStringArrayListFunc, key, value);
}

jbyte JavaBundle::getByte(jstring key) {
    if (Bundle_getByteFunc == nullptr) {
        Bundle_getByteFunc = mJNIEnv->GetMethodID(Bundle_Class, "getByte", "(Ljava/lang/String;)B");
    }
    return mJNIEnv->CallByteMethod(mJavaBundleObj, Bundle_getByteFunc, key);
}

void JavaBundle::putByte(jstring key, jbyte value) {
    if (Bundle_putByteFunc == nullptr) {
        Bundle_putByteFunc = mJNIEnv->GetMethodID(Bundle_Class, "putByte",
                                                  "(Ljava/lang/String;B)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putByteFunc, key, value);
}

jbyteArray JavaBundle::getByteArray(jstring key) {
    if (Bundle_getByteArrayFunc == nullptr) {
        Bundle_getByteArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getByteArray",
                                                       "(Ljava/lang/String;)[B");
    }
    return (jbyteArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getByteArrayFunc, key);
}

void JavaBundle::putByteArray(jstring key, jbyteArray value) {
    if (Bundle_putByteArrayFunc == nullptr) {
        Bundle_putByteArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putByteArray",
                                                       "(Ljava/lang/String;[B)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putByteArrayFunc, key, value);
}

jchar JavaBundle::getChar(jstring key) {
    if (Bundle_getCharFunc == nullptr) {
        Bundle_getCharFunc = mJNIEnv->GetMethodID(Bundle_Class, "getChar", "(Ljava/lang/String;)C");
    }
    return mJNIEnv->CallCharMethod(mJavaBundleObj, Bundle_getCharFunc, key);
}

void JavaBundle::putChar(jstring key, jchar value) {
    if (Bundle_putCharFunc == nullptr) {
        Bundle_putCharFunc = mJNIEnv->GetMethodID(Bundle_Class, "putChar",
                                                  "(Ljava/lang/String;C)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putCharFunc, key, value);
}

jcharArray JavaBundle::getCharArray(jstring key) {
    if (Bundle_getCharArrayFunc == nullptr) {
        Bundle_getCharArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getCharArray",
                                                       "(Ljava/lang/String;)[C");
    }
    return (jcharArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getCharArrayFunc, key);
}

void JavaBundle::putCharArray(jstring key, jcharArray value) {
    if (Bundle_putCharArrayFunc == nullptr) {
        Bundle_putCharArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putCharArray",
                                                       "(Ljava/lang/String;[C)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putCharArrayFunc, key, value);
}

jfloat JavaBundle::getFloat(jstring key) {
    if (Bundle_getFloatFunc == nullptr) {
        Bundle_getFloatFunc = mJNIEnv->GetMethodID(Bundle_Class, "getFloat",
                                                   "(Ljava/lang/String;)F");
    }
    return mJNIEnv->CallFloatMethod(mJavaBundleObj, Bundle_getFloatFunc, key);
}

void JavaBundle::putFloat(jstring key, jfloat value) {
    if (Bundle_putFloatFunc == nullptr) {
        Bundle_putFloatFunc = mJNIEnv->GetMethodID(Bundle_Class, "putFloat",
                                                   "(Ljava/lang/String;F)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putFloatFunc, key, value);
}

jfloatArray JavaBundle::getFloatArray(jstring key) {
    if (Bundle_getFloatArrayFunc == nullptr) {
        Bundle_getFloatArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getFloatArray",
                                                        "(Ljava/lang/String;)[F");
    }
    return (jfloatArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getFloatArrayFunc, key);
}

void JavaBundle::putFloatArray(jstring key, jfloatArray value) {
    if (Bundle_putFloatArrayFunc == nullptr) {
        Bundle_putFloatArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putFloatArray",
                                                        "(Ljava/lang/String;[F)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putFloatArrayFunc, key, value);
}

jlong JavaBundle::getLong(jstring key) {
    if (Bundle_getLongFunc == nullptr) {
        Bundle_getLongFunc = mJNIEnv->GetMethodID(Bundle_Class, "getLong", "(Ljava/lang/String;)J");
    }
    return mJNIEnv->CallLongMethod(mJavaBundleObj, Bundle_getLongFunc, key);
}

void JavaBundle::putLong(jstring key, jlong value) {
    if (Bundle_putLongFunc == nullptr) {
        Bundle_putLongFunc = mJNIEnv->GetMethodID(Bundle_Class, "putLong",
                                                  "(Ljava/lang/String;J)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putLongFunc, key, value);
}

jlongArray JavaBundle::getLongArray(jstring key) {
    if (Bundle_getLongArrayFunc == nullptr) {
        Bundle_getLongArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getLongArray",
                                                       "(Ljava/lang/String;)[J");
    }
    return (jlongArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getLongArrayFunc, key);
}

void JavaBundle::putLongArray(jstring key, jlongArray value) {
    if (Bundle_putLongArrayFunc == nullptr) {
        Bundle_putLongArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putLongArray",
                                                       "(Ljava/lang/String;[J)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putLongArrayFunc, key, value);
}

jobject JavaBundle::getBundle(jstring key) {
    if (Bundle_getBundleFunc == nullptr) {
        Bundle_getBundleFunc = mJNIEnv->GetMethodID(Bundle_Class, "getBundle",
                                                    "(Ljava/lang/String;)Landroid/os/Bundle;");
    }
    return mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getBundleFunc, key);
}

void JavaBundle::putBundle(jstring key, jobject value) {
    if (Bundle_putBundleFunc == nullptr) {
        Bundle_putBundleFunc = mJNIEnv->GetMethodID(Bundle_Class, "putBundle",
                                                    "(Ljava/lang/String;Landroid/os/Bundle;)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putBundleFunc, key, value);
}

jobjectArray JavaBundle::getParcelableArray(jstring key) {
    if (Bundle_getParcelableArrayFunc == nullptr) {
        Bundle_getParcelableArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "getParcelableArray",
                                                             "(Ljava/lang/String;)[Landroid/os/Parcelable;");
    }
    return (jobjectArray) mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getParcelableArrayFunc,
                                                    key);
}

void JavaBundle::putParcelableArray(jstring key, jobjectArray value) {
    if (Bundle_putParcelableArrayFunc == nullptr) {
        Bundle_putParcelableArrayFunc = mJNIEnv->GetMethodID(Bundle_Class, "putParcelableArray",
                                                             "(Ljava/lang/String;[Landroid/os/Parcelable;)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putParcelableArrayFunc, key, value);
}

jobject JavaBundle::getParcelableArrayList(jstring key) {
    if (Bundle_getParcelableArrayListFunc == nullptr) {
        Bundle_getParcelableArrayListFunc = mJNIEnv->GetMethodID(Bundle_Class,
                                                                 "getParcelableArrayList",
                                                                 "(Ljava/lang/String;)Ljava/util/ArrayList;");
    }
    return mJNIEnv->CallObjectMethod(mJavaBundleObj, Bundle_getParcelableArrayListFunc, key);
}

void JavaBundle::putParcelableArrayList(jstring key, jobject value) {
    if (Bundle_putParcelableArrayListFunc == nullptr) {
        Bundle_putParcelableArrayListFunc = mJNIEnv->GetMethodID(Bundle_Class,
                                                                 "putParcelableArrayList",
                                                                 "(Ljava/lang/String;Ljava/util/ArrayList;)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_putParcelableArrayListFunc, key, value);
}

void JavaBundle::clear() {
    if (Bundle_clearFunc == nullptr) {
        Bundle_clearFunc = mJNIEnv->GetMethodID(Bundle_Class, "clear", "()V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_clearFunc);
}

void JavaBundle::remove(jstring key) {
    if (Bundle_removeFunc == nullptr) {
        Bundle_removeFunc = mJNIEnv->GetMethodID(Bundle_Class, "remove", "(Ljava/lang/String;)V");
    }
    mJNIEnv->CallVoidMethod(mJavaBundleObj, Bundle_removeFunc, key);
}

jboolean JavaBundle::containsKey(jstring key) {
    if (Bundle_containsKeyFunc == nullptr) {
        Bundle_containsKeyFunc = mJNIEnv->GetMethodID(Bundle_Class, "containsKey",
                                                      "(Ljava/lang/String;)Z");
    }
    return mJNIEnv->CallBooleanMethod(mJavaBundleObj, Bundle_containsKeyFunc, key);
}

jint JavaBundle::getInt(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jint value = getInt(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putInt(const char *key, jint value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putInt(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jintArray JavaBundle::getIntArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jintArray value = getIntArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putIntArray(const char *key, jintArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putIntArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jboolean JavaBundle::getBoolean(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jboolean value = getBoolean(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putBoolean(const char *key, jboolean value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putBoolean(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jbooleanArray JavaBundle::getBooleanArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jbooleanArray value = getBooleanArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putBooleanArray(const char *key, jbooleanArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putBooleanArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jshort JavaBundle::getShort(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jshort value = getShort(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putShort(const char *key, jshort value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putShort(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jshortArray JavaBundle::getShortArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jshortArray value = getShortArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putShortArray(const char *key, jshortArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putShortArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jdouble JavaBundle::getDouble(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jdouble value = getDouble(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putDouble(const char *key, jdouble value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putDouble(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jdoubleArray JavaBundle::getDoubleArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jdoubleArray value = getDoubleArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putDoubleArray(const char *key, jdoubleArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putDoubleArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jstring JavaBundle::getJString(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jstring value = getJString(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putJString(const char *key, jstring value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putJString(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jobjectArray JavaBundle::getStringArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jobjectArray value = getStringArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putStringArray(const char *key, jobjectArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putStringArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jobject JavaBundle::getStringArrayList(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jobject value = getStringArrayList(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putStringArrayList(const char *key, jobject value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putStringArrayList(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jbyte JavaBundle::getByte(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jbyte value = getByte(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putByte(const char *key, jbyte value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putByte(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jbyteArray JavaBundle::getByteArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jbyteArray value = getByteArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putByteArray(const char *key, jbyteArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putByteArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jchar JavaBundle::getChar(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jchar value = getChar(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putChar(const char *key, jchar value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putChar(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jcharArray JavaBundle::getCharArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jcharArray value = getCharArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putCharArray(const char *key, jcharArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putCharArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jfloat JavaBundle::getFloat(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jfloat value = getFloat(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putFloat(const char *key, jfloat value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putFloat(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jfloatArray JavaBundle::getFloatArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jfloatArray value = getFloatArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putFloatArray(const char *key, jfloatArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putFloatArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jlong JavaBundle::getLong(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jlong value = getLong(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putLong(const char *key, jlong value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putLong(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jlongArray JavaBundle::getLongArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jlongArray value = getLongArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putLongArray(const char *key, jlongArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putLongArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jobject JavaBundle::getBundle(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jobject value = getBundle(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putBundle(const char *key, jobject value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putBundle(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jobjectArray JavaBundle::getParcelableArray(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jobjectArray value = getParcelableArray(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putParcelableArray(const char *key, jobjectArray value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putParcelableArray(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}

jobject JavaBundle::getParcelableArrayList(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jobject value = getParcelableArrayList(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putParcelableArrayList(const char *key, jobject value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    putParcelableArrayList(jkey, value);
    mJNIEnv->DeleteLocalRef(jkey);
}


void JavaBundle::remove(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    remove(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
}

jboolean JavaBundle::containsKey(const char *key) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jboolean value = containsKey(jkey);
    mJNIEnv->DeleteLocalRef(jkey);
    return value;
}

void JavaBundle::putString(const char *key, std::string &value) {
    assertKey(key);
    jstring jkey = mJNIEnv->NewStringUTF(key);
    jstring jvalue = mJNIEnv->NewStringUTF(value.data());
    putJString(jkey, jvalue);
    mJNIEnv->DeleteLocalRef(jkey);
    mJNIEnv->DeleteLocalRef(jvalue);
}

void JavaBundle::putString(jstring key, std::string &value) {
    jstring jValue = mJNIEnv->NewStringUTF(value.data());
    putJString(key, jValue);
    mJNIEnv->DeleteLocalRef(jValue);
}

std::string JavaBundle::getString(const char *key) {
    jstring jvalue = getJString(key);
    if (jvalue == nullptr) {
        return "";
    }
    const char *chars = mJNIEnv->GetStringUTFChars(jvalue, 0);
    std::string value = chars;
    mJNIEnv->ReleaseStringUTFChars(jvalue, chars);
    mJNIEnv->DeleteLocalRef(jvalue);
    return value;
}

std::string JavaBundle::getString(jstring key) {
    jstring jvalue = getJString(key);
    if (jvalue == nullptr) {
        return nullptr;
    }
    const char *chars = mJNIEnv->GetStringUTFChars(jvalue, 0);
    std::string value = chars;
    mJNIEnv->ReleaseStringUTFChars(jvalue, chars);
    mJNIEnv->DeleteLocalRef(jvalue);
    return value;
}

void JavaBundle::ensureBundleClassNotNull() {
    if (Bundle_Class == nullptr) {
        JavaVM *vm;
        JNIEnv *jniEnv;
        jint jniVersion;
        vm = JVMContainer::GetJVM();
        jniVersion = JVMContainer::GetJvmVersion();
        if (vm->GetEnv((void **) &jniEnv, jniVersion) != JNI_OK) {
            return;
        }
        jclass localRef = (jclass) jniEnv->FindClass(JAVA_BUNDLE_CLASS_NAME);
        if (localRef) {
            Bundle_Class = (jclass) jniEnv->NewGlobalRef(localRef);
            jniEnv->DeleteLocalRef(localRef);
        }
    }
}

jobject JavaBundle::getJavaBundleObject() {
    return mJavaBundleObj;
}


void JavaBundle::assertKey(const void *key) {
    if (key == nullptr) {
        mJNIEnv->ThrowNew(mJNIEnv->FindClass("java/lang/Exception"),
                          "JavaBundle::assertKey key is nullptr");
    }
}