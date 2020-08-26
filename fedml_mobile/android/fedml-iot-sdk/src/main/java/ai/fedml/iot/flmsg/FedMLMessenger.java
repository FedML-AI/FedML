package ai.fedml.iot.flmsg;

import android.content.Context;
import ai.fedml.iot.Config;
import ai.fedml.iot.mq.FLMessage;
import ai.fedml.iot.mq.MQManager;

public class FedMLMessenger {
    private static final String TAG = Config.COMMON_TAG + FedMLMessenger.class.getSimpleName();

    private volatile static FedMLMessenger mInstance = null;

    public static FedMLMessenger getInstance() {
        if (mInstance == null) {
            synchronized (FedMLMessenger.class) {
                if (mInstance == null) {
                    mInstance = new FedMLMessenger();
                }
            }
        }
        return mInstance;
    }


    public static synchronized void destroy() {
        if (mInstance != null) {
            mInstance.unInit();
        }
        mInstance = null;
    }

    private Context mContext;

    private MQManager mMQManager = null;

    private FedMLMessenger() {
    }

    public void init(Context context, MQManager sender) {
        mContext = context;
        mMQManager = sender;
    }

    private void unInit(){
        mMQManager = null;
    }

    private void sendLocation(FLMessage flmsg) {
        mMQManager.sendLocation(flmsg);
    }
}
