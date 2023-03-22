package ai.fedml.edge.utils

import android.os.Build
import android.text.TextUtils
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.io.RandomAccessFile

object CpuUtil {
    private var mProcStatFile: RandomAccessFile? = null
    private var mAppStatFile: RandomAccessFile? = null
    private var mLastCpuTime: Long? = null
    private var mLastAppCpuTime: Long? = null

    /**
     * get Cpu Usage
     */
    fun getCpuUsage(): Float {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getCpuUsageForHigherVersion()
        } else {
            getCpuUsageForLowerVersion()
        }
    }

    private fun getCpuUsageForHigherVersion(): Float {

        var process: Process? = null
        try {

            process = Runtime.getRuntime().exec("top -n 1")
            val reader = BufferedReader(InputStreamReader(process.inputStream))
            var line: String
            var cpuIndex = -1
            while (reader.readLine().also {
                        line = it
                    } != null) {

                line = line.trim {
                    it <= ' '
                }
                if (TextUtils.isEmpty(line)) {
                    continue
                }
                val tempIndex = getCPUIndex(line)
                if (tempIndex != -1) {

                    cpuIndex = tempIndex
                    continue
                }
                if (line.startsWith(android.os.Process.myPid().toString())) {

                    if (cpuIndex == -1) {

                        continue
                    }
                    val param = line.split("\\s+".toRegex()).toTypedArray()
                    if (param.size <= cpuIndex) {

                        continue
                    }
                    var cpu = param[cpuIndex]
                    if (cpu.endsWith("%")) {

                        cpu = cpu.substring(0, cpu.lastIndexOf("%"))
                    }
                    return cpu.toFloat() / Runtime.getRuntime().availableProcessors()
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        } finally {
            process?.destroy()
        }
        return 0F
    }

    private fun getCpuUsageForLowerVersion(): Float {
        val cpuTime: Long
        val appTime: Long
        var value = 0.0f
        try {
            if (mProcStatFile == null || mAppStatFile == null) {
                mProcStatFile = RandomAccessFile("/proc/stat", "r")
                mAppStatFile = RandomAccessFile("/proc/" + android.os.Process.myPid() + "/stat", "r")
            } else {
                mProcStatFile!!.seek(0L)
                mAppStatFile!!.seek(0L)
            }
            val procStatString = mProcStatFile!!.readLine()
            val appStatString = mAppStatFile!!.readLine()
            val procStats = procStatString.split(" ".toRegex()).toTypedArray()
            val appStats = appStatString.split(" ".toRegex()).toTypedArray()
            cpuTime = procStats[2].toLong() + procStats[3].toLong() + procStats[4].toLong() + procStats[5].toLong() + procStats[6].toLong() + procStats[7].toLong() + procStats[8].toLong()
            appTime = appStats[13].toLong() + appStats[14].toLong()
            if (mLastCpuTime == null && mLastAppCpuTime == null) {

                mLastCpuTime = cpuTime
                mLastAppCpuTime = appTime
                return value
            }
            value = (appTime - mLastAppCpuTime!!).toFloat() / (cpuTime - mLastCpuTime!!).toFloat() * 100f
            mLastCpuTime = cpuTime
            mLastAppCpuTime = appTime
        } catch (e: Exception) {

            e.printStackTrace()
        }
        return value
    }

    private fun getCPUIndex(line: String): Int {
        if (line.contains("CPU")) {

            val titles = line.split("\\s+".toRegex()).toTypedArray()
            for (i in titles.indices) {

                if (titles[i].contains("CPU")) {
                    return i
                }
            }
        }
        return -1
    }
}