package ai.fedml.edge.utils;

import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;

import java.util.concurrent.atomic.AtomicBoolean;

public class BackgroundHandler extends Handler {
    public BackgroundHandler(String name) {
        this(name, 1);
    }

    public BackgroundHandler(String name, int priority) {
        super(startHandlerThread(name, priority).getLooper());
    }

    private static HandlerThread startHandlerThread(String name,
                                                    int priority) {
        final AtomicBoolean localAtomicBoolean = new AtomicBoolean(false);
        HandlerThread ht = new HandlerThread(name, priority) {
            protected void onLooperPrepared() {
                localAtomicBoolean.compareAndSet(false, true);
            }
        };
        ht.start();
        while (!localAtomicBoolean.get()) ;
        return ht;
    }

    public void dispatchMessage(Message msg) {
        try{
            super.dispatchMessage(msg);
        }catch(Throwable e){
            e.printStackTrace();
        }
    }

    public void quit() {
        getLooper().quit();
    }

    public boolean sendMessageAtTime(Message msg, long delay) {
        return super.sendMessageAtTime(msg, delay);
    }
}
