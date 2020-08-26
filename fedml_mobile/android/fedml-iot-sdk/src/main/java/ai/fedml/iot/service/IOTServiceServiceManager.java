package ai.fedml.iot.service;

import android.content.ComponentName;
import android.os.IBinder;
import android.os.RemoteException;

import ai.fedml.iot.IMQService;

/**
 * IOTServiceServiceManager -> BaseServiceManager -> bind IOTService -> IOTServiceImpl
 */
public class IOTServiceServiceManager extends BaseServiceManager {

    private static final String TAG = IOTServiceServiceManager.class.getSimpleName();
    private static volatile IOTServiceServiceManager mInstance;

    private IMQService mMQService;

    @Override
    protected void onServiceConnectedOk(IBinder service) {
        mMQService = IMQService.Stub.asInterface(service);
        try {
            mMQService.onServiceConnectedOk();
        } catch (RemoteException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onServiceDisconnected(ComponentName name) {
        try {
            mMQService.onServiceDisconnected();
            mMQService = null;
        } catch (RemoteException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void dispose() {

    }

    public static IOTServiceServiceManager getInstance() {
        if (null == mInstance) {
            synchronized (IOTServiceServiceManager.class) {
                if (null == mInstance) {
                    mInstance = new IOTServiceServiceManager();
                }
            }
        }
        return mInstance;
    }

    public int processMessage(int command){
        try {
            if(mMQService != null){
                return mMQService.processMessage(command);
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        return 0;
    }

    private IOTServiceServiceManager() {
        super(IOTService.ACTION_MQTT_SERVICE);
    }
}
