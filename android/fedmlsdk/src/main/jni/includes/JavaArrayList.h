#ifndef _JAVA_ARRAYLIST_H_
#define _JAVA_ARRAYLIST_H_
#include <jni.h>
#include <cstddef>

 class JavaArrayList
 {
 public:
       JavaArrayList(JNIEnv* jniEnv);
       JavaArrayList(JNIEnv* jniEnv,jobject javaArrayListObject);
       ~JavaArrayList();
       jint size();
       void clear();
       void add(jint index,jobject javaObject);
       void add(jobject javaObject);
       void addAll(jobject javaObject);
       jboolean remove(jobject javaObject);
       jobject remove(jint index);
       jobject get(jint index);
       jobject getJavaArrayListObject();
       jboolean contains(jobject javaObject);
       static void ensureArrayListClassNotNull();

 private:
        JNIEnv* mJNIEnv;
        jobject mJavaArrayListObj;

 };


 #endif

