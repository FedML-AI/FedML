#ifndef _JAVA_BUNDLE_H_
#define _JAVA_BUNDLE_H_
#include <jni.h>
#include <cstddef>
#include <string>

class JavaBundle
 {
 public:
       JavaBundle(JNIEnv* jniEnv);
       JavaBundle(JNIEnv* jniEnv,jobject javaBundleObject);
       ~JavaBundle();
       jint                 getInt(jstring key);
       void                 putInt(jstring key, jint value);
       jintArray            getIntArray(jstring key);
       void                 putIntArray(jstring key, jintArray value);

       jboolean             getBoolean(jstring key);
       void                 putBoolean(jstring key, jboolean value);
       jbooleanArray        getBooleanArray(jstring key);
       void                 putBooleanArray(jstring key, jbooleanArray value);

       jshort               getShort(jstring key);
       void                 putShort(jstring key, jshort value);
       jshortArray          getShortArray(jstring key);
       void                 putShortArray(jstring key, jshortArray value);

       jdouble              getDouble(jstring key);
       void                 putDouble(jstring key, jdouble value);
       jdoubleArray         getDoubleArray(jstring key);
       void                 putDoubleArray(jstring key, jdoubleArray value);

       jstring              getJString(jstring key);
       void                 putJString(jstring key, jstring value);
       jobjectArray         getStringArray(jstring key);
       void                 putStringArray(jstring key, jobjectArray value);
       jobject              getStringArrayList(jstring key);
       void                 putStringArrayList(jstring key, jobject value);

       jbyte                getByte(jstring key);
       void                 putByte(jstring key, jbyte value);
       jbyteArray           getByteArray(jstring key);
       void                 putByteArray(jstring key, jbyteArray value);

       jchar                getChar(jstring key);
       void                 putChar(jstring key, jchar value);
       jcharArray           getCharArray(jstring key);
       void                 putCharArray(jstring key, jcharArray value);

       jfloat               getFloat(jstring key);
       void                 putFloat(jstring key, jfloat value);
       jfloatArray          getFloatArray(jstring key);
       void                 putFloatArray(jstring key, jfloatArray value);

       jlong                getLong(jstring key);
       void                 putLong(jstring key, jlong value);
       jlongArray           getLongArray(jstring key);
       void                 putLongArray(jstring key, jlongArray value);

       jobject              getBundle(jstring key);
       void                 putBundle(jstring key, jobject value);

       jobjectArray         getParcelableArray(jstring key);
       void                 putParcelableArray(jstring key, jobjectArray value);
       jobject              getParcelableArrayList(jstring key);
       void                 putParcelableArrayList(jstring key, jobject value);

       void                 clear();
       void                 remove(jstring key);
       jboolean             containsKey(jstring key);
       jobject              getJavaBundleObject();

       jint                 getInt(const char* key);
       void                 putInt(const char* key, jint value);
       jintArray            getIntArray(const char* key);
       void                 putIntArray(const char* key, jintArray value);

       jboolean             getBoolean(const char* key);
       void                 putBoolean(const char* key, jboolean value);
       jbooleanArray        getBooleanArray(const char* key);
       void                 putBooleanArray(const char* key, jbooleanArray value);

       jshort               getShort(const char* key);
       void                 putShort(const char* key, jshort value);
       jshortArray          getShortArray(const char* key);
       void                 putShortArray(const char* key, jshortArray value);

       jdouble              getDouble(const char* key);
       void                 putDouble(const char* key, jdouble value);
       jdoubleArray         getDoubleArray(const char* key);
       void                 putDoubleArray(const char* key, jdoubleArray value);

       jstring              getJString(const char* key);
       void                 putJString(const char* key, jstring value);
       jobjectArray         getStringArray(const char* key);
       void                 putStringArray(const char* key, jobjectArray value);
       jobject              getStringArrayList(const char* key);
       void                 putStringArrayList(const char* key, jobject value);

       jbyte                getByte(const char* key);
       void                 putByte(const char* key, jbyte value);
       jbyteArray           getByteArray(const char* key);
       void                 putByteArray(const char* key, jbyteArray value);

       jchar                getChar(const char* key);
       void                 putChar(const char* key, jchar value);
       jcharArray           getCharArray(const char* key);
       void                 putCharArray(const char* key, jcharArray value);

       jfloat               getFloat(const char* key);
       void                 putFloat(const char* key, jfloat value);
       jfloatArray          getFloatArray(const char* key);
       void                 putFloatArray(const char* key, jfloatArray value);

       jlong                getLong(const char* key);
       void                 putLong(const char* key, jlong value);
       jlongArray           getLongArray(const char* key);
       void                 putLongArray(const char* key, jlongArray value);

       jobject              getBundle(const char* key);
       void                 putBundle(const char* key, jobject value);

       jobjectArray         getParcelableArray(const char* key);
       void                 putParcelableArray(const char* key, jobjectArray value);
       jobject              getParcelableArrayList(const char* key);
       void                 putParcelableArrayList(const char* key, jobject value);

       void                 remove(const char* key);
       jboolean             containsKey(const char* key);

       void                 putString(const char* key, std::string& value);
       void                 putString(jstring key, std::string& value);
       std::string          getString(const char* key);
       std::string          getString(jstring key);
       static void ensureBundleClassNotNull();
 private:
       void assertKey(const void* key);
 private:
        JNIEnv* mJNIEnv;
        jobject mJavaBundleObj;

 };


 #endif  // _JBUNDLE_H_

