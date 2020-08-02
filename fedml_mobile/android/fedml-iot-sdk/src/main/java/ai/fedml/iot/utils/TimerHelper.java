package ai.fedml.iot.utils;


import android.os.SystemClock;

public class TimerHelper {

    private long mCurrentTime;
    private long mElapsedRealtime;
    private static final Object mMutext = new Object();
    public TimerHelper(long time) {
        mElapsedRealtime = SystemClock.elapsedRealtime();
        mCurrentTime = time;
    }


    public void setCurrentTime(long time) {
        synchronized (mMutext) {
            mElapsedRealtime = SystemClock.elapsedRealtime();
            mCurrentTime = time;
        }

    }

    public long getCurrentTime() {
        synchronized (mMutext) {
            return (SystemClock.elapsedRealtime() - mElapsedRealtime) + mCurrentTime;
        }
    }

}
