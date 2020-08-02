#!/usr/bin/env bash
export JDK_HOME=/Applications/Android\ Studio.app/Contents/jre/jdk/Contents/Home
export GRADLE_HOME=/Applications/Android\ Studio.app/Contents/plugins/gradle/lib
export PATH=$GRADLE_HOME:$PATH
export PATH=$JDK_HOME/bin:$PATH

echo GRADLE_HOME=$GRADLE_HOME
echo JDK_HOME=$GRADLE_HOME
gradle makeJar