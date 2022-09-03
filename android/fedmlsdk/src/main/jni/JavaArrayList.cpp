#include "includes/JavaArrayList.h"
#include "includes/JVMContainer.h"
#include <cassert>
static jclass    List_Class = nullptr;
static jclass    ArrayList_Class = nullptr;
jmethodID ArrayList_constructorFunc = nullptr;
jmethodID ArrayList_getFunc = nullptr;
jmethodID ArrayList_addFunc = nullptr;
jmethodID ArrayList_addIndexFunc = nullptr;
jmethodID ArrayList_addAllFunc = nullptr;
jmethodID ArrayList_removeIndexFunc = nullptr;
jmethodID ArrayList_removeObjFunc = nullptr;
jmethodID ArrayList_containsFunc = nullptr;
jmethodID ArrayList_sizeFunc = nullptr;
jmethodID ArrayList_clearFunc = nullptr;



#define JAVA_List_Class_NAME "java/util/List"
#define JAVA_ArrayList_Class_NAME "java/util/ArrayList"

JavaArrayList::JavaArrayList(JNIEnv* jniEnv)
{
    mJNIEnv = jniEnv;
    assert(mJNIEnv);

    JavaArrayList::ensureArrayListClassNotNull();
    if (ArrayList_constructorFunc == nullptr)
    {
        ArrayList_constructorFunc = mJNIEnv->GetMethodID(ArrayList_Class,"<init>", "()V");
    }
    mJavaArrayListObj = mJNIEnv->NewObject(ArrayList_Class,ArrayList_constructorFunc);
}
JavaArrayList::JavaArrayList(JNIEnv* jniEnv,jobject javaArrayListObject)
{
    mJNIEnv = jniEnv;
    assert(mJNIEnv);

    if (javaArrayListObject == nullptr)
    {
       mJNIEnv->ThrowNew(mJNIEnv->FindClass("java/lang/Exception"),"JavaArrayList::JavaArrayList---javaArrayListObject is nullptr");
    }
    JavaArrayList::ensureArrayListClassNotNull();
    mJavaArrayListObj = javaArrayListObject;
}

JavaArrayList::~JavaArrayList()
{
    mJNIEnv->DeleteLocalRef(mJavaArrayListObj);
    mJNIEnv = nullptr;
}

jobject  JavaArrayList::get(jint index)
{
    if (ArrayList_getFunc == nullptr)
    {
       ArrayList_getFunc = mJNIEnv->GetMethodID(List_Class,"get","(I)Ljava/lang/Object;");
    }
    return mJNIEnv->CallObjectMethod(mJavaArrayListObj,ArrayList_getFunc,index);
}

void  JavaArrayList::JavaArrayList::add(jobject object)
{
    if (ArrayList_addFunc == nullptr)
    {
       ArrayList_addFunc = mJNIEnv->GetMethodID(List_Class,"add","(Ljava/lang/Object;)Z");
    }
    mJNIEnv->CallBooleanMethod(mJavaArrayListObj,ArrayList_addFunc,object);
}

void  JavaArrayList::JavaArrayList::add(jint index,jobject object)
{
    if (ArrayList_addIndexFunc == nullptr)
    {
       ArrayList_addIndexFunc = mJNIEnv->GetMethodID(List_Class,"add","(ILjava/lang/Object;)V");
    }
    mJNIEnv->CallVoidMethod(mJavaArrayListObj,ArrayList_addIndexFunc,index,object);
}

void  JavaArrayList::JavaArrayList::addAll(jobject object)
{
    if (ArrayList_addAllFunc == nullptr)
    {
       ArrayList_addAllFunc = mJNIEnv->GetMethodID(List_Class,"addAll","(Ljava/util/Collection;)Z");
    }
    mJNIEnv->CallBooleanMethod(mJavaArrayListObj,ArrayList_addAllFunc,object);
}

jobject   JavaArrayList::remove(jint index)
{
    if (ArrayList_removeIndexFunc == nullptr)
    {
       ArrayList_removeIndexFunc = mJNIEnv->GetMethodID(List_Class,"remove","(I)Ljava/lang/Object;");
    }
    return mJNIEnv->CallObjectMethod(mJavaArrayListObj,ArrayList_removeIndexFunc,index);
}

jboolean  JavaArrayList::remove(jobject object)
{
    if (ArrayList_removeObjFunc == nullptr)
    {
       ArrayList_removeObjFunc = mJNIEnv->GetMethodID(List_Class,"remove","(Ljava/lang/Object;)Z");
    }
    return mJNIEnv->CallBooleanMethod(mJavaArrayListObj,ArrayList_removeObjFunc,object);
}

jboolean   JavaArrayList::contains(jobject object)
{
    if (ArrayList_containsFunc == nullptr)
    {
       ArrayList_containsFunc = mJNIEnv->GetMethodID(List_Class,"contains","(Ljava/lang/Object;)Z");
    }
    return mJNIEnv->CallBooleanMethod(mJavaArrayListObj,ArrayList_containsFunc,object);
}

jint   JavaArrayList::size()
{
    if (ArrayList_sizeFunc == nullptr)
    {
       ArrayList_sizeFunc = mJNIEnv->GetMethodID(List_Class,"size","()I");
    }
    return mJNIEnv->CallIntMethod(mJavaArrayListObj,ArrayList_sizeFunc);
}

void   JavaArrayList::clear()
{
    if (ArrayList_clearFunc == nullptr)
    {
       ArrayList_clearFunc = mJNIEnv->GetMethodID(List_Class,"clear","()V");
    }
    return mJNIEnv->CallVoidMethod(mJavaArrayListObj,ArrayList_clearFunc);
}


void JavaArrayList::ensureArrayListClassNotNull()
{
    if (List_Class == nullptr)
    {
        JavaVM* vm;
        JNIEnv* jniEnv;
        jint    jniVersion;
        vm = JVMContainer::GetJVM();
        jniVersion = JVMContainer::GetJvmVersion();
        if (vm->GetEnv((void **)&jniEnv,jniVersion) != JNI_OK)
        {
            return;
        }
        auto localRef = (jclass)jniEnv->FindClass(JAVA_List_Class_NAME);
        if (localRef)
        {
            List_Class = (jclass)jniEnv->NewGlobalRef(localRef);
            jniEnv->DeleteLocalRef(localRef);
        }

        auto localArrayListRef = (jclass)jniEnv->FindClass(JAVA_ArrayList_Class_NAME);
        if (localArrayListRef)
        {
            ArrayList_Class = (jclass)jniEnv->NewGlobalRef(localArrayListRef);
            jniEnv->DeleteLocalRef(localArrayListRef);
        }
    }
}

jobject  JavaArrayList::getJavaArrayListObject()
{
     return mJavaArrayListObj;
}