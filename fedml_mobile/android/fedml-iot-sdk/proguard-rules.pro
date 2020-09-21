# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in D:\program\Android\android-sdk-windows/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Add any project specific keep options here:

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile
-ignorewarnings
-dontoptimize
-dontpreverify
-keepattributes Exceptions,InnerClasses,Signature,Deprecated,SourceFile,LineNumberTable,*Annotation*,Synthetic,EnclosingMethod

-keep class * extends android.app.Activity
-keep class * extends android.app.Application
-keep class * extends android.app.Service
-keep class * extends android.content.BroadcastReceiver
-keep class * extends android.content.ContentProvider
-keep class * extends android.app.backup.BackupAgentHelper
-keep class * extends android.preference.Preference
-keep class * extends android.os.Bundle

-dontwarn com.tencent.bugly.**
-keep public class com.tencent.bugly.**{*;}

-keep class ai.fedml.iot.** { *; }
#-keep class ai.fedml.iovcore.config.SDKConfig { *; }
-dontwarn ai.fedml.pluginiov.device.VehicleDevice


#native jni接口
-keepclasseswithmembernames class * {
    native <methods>;
}

-keepattributes SourceFile,LineNumberTable

##-- keep Parcelable Object
-keep class * implements android.os.Parcelable {
  public static final android.os.Parcelable$Creator *;
}

# sdk版本小于18时需要以下配置, 建议使用18或以上版本的sdk编译
-dontwarn  android.location.Location
-keep class com.tencent.map.**{*;}
-keep interface com.tencent.map.**{*;}
-dontwarn com.tencent.map.**
-keep class com.tencent.tencentmap.**{*;}
-dontwarn com.tencent.tencentmap.**
-keep class ct.**{*;}
-keep interface ct.**{*;}
-dontwarn ct.**
-keep class c.t.**{*;}
-keep interface c.t.**{*;}
-dontwarn c.t.**
-keepclassmembers class ** {
    public void on*Event(...);
}

-dontwarn  org.eclipse.jdt.annotation.**

-keep class org.apache.commons.codec.**{*;}
-keep interface org.apache.commons.codec.**{*;}
-dontwarn org.apache.commons.codec.**

-dontwarn com.tencent.bugly.**
-keep public class com.tencent.bugly.**{*;}