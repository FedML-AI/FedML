package ai.fedml.edge.service;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;

import ai.fedml.edge.EdgeMessageDefine;
import ai.fedml.edge.OnAccuracyLossListener;
import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.service.entity.TrainProgress;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class EdgeService extends Service implements EdgeMessageDefine {
    private final static int GRAY_SERVICE_ID = 1001;
    private static final String NOTIFY_CHANNEL_ID = "Channel_Id";
    private final FedEdgeTrainApi fedEdgeTrainApi = new FedEdgeTrainImpl();
    private final OnTrainProgressListener onAccuracyLossListener = new OnTrainProgressListener() {
        @Override
        public void onEpochLoss(int round, int epoch, float loss) {
            Message message = Message.obtain();
            message.what = MSG_TRAIN_LOSS;
            message.arg1 = round;
            message.arg2 = epoch;
            Bundle bundle = new Bundle();
            bundle.putInt(TRAIN_EPOCH, epoch);
            bundle.putFloat(TRAIN_LOSS, loss);
            message.setData(bundle);
            sendMessageToClient(message);
        }

        @Override
        public void onEpochAccuracy(int round, int epoch, float accuracy) {
            Message message = Message.obtain();
            message.what = MSG_TRAIN_ACCURACY;
            message.arg1 = round;
            message.arg2 = epoch;
            Bundle bundle = new Bundle();
            bundle.putInt(TRAIN_EPOCH, epoch);
            bundle.putFloat(TRAIN_ACCURACY, accuracy);
            message.setData(bundle);
            sendMessageToClient(message);
        }

        @Override
        public void onProgressChanged(int round, float progress) {
            Message message = Message.obtain();
            message.what = MSG_TRAIN_PROGRESS;
            message.arg1 = round;
            message.arg2 = (int)progress;
            sendMessageToClient(message);
        }
    };
    private final OnTrainingStatusListener onTrainingStatusListener = status -> {
        Message message = Message.obtain();
        message.what = MSG_TRAIN_STATUS;
        message.arg1 = status;
        sendMessageToClient(message);
    };

    private Messenger mClientMessenger;
    private final Handler serviceHandler = new Handler(Looper.getMainLooper()) {

        @Override
        public void handleMessage(Message msg) {
            LogHelper.d("receive message from client:%d", msg.what);
            LogHelper.d("ClientMessenger=" + msg.replyTo);
            mClientMessenger = msg.replyTo;
            if (msg.what == MSG_START_TRAIN) {
                LogHelper.d("receive message from client:%s", msg.getData().getString(TRAIN_ARGS));
            } else if (msg.what == MSG_TRAIN_STATUS) {
                Message message = Message.obtain();
                message.what = MSG_TRAIN_STATUS;
                message.arg1 = fedEdgeTrainApi.getTrainStatus();
                callbackMessage(msg.replyTo, message);
            } else if (MSG_TRAIN_PROGRESS == msg.what) {
                Message message = Message.obtain();
                message.what = MSG_TRAIN_PROGRESS;
                TrainProgress progress = fedEdgeTrainApi.getTrainProgress();
                message.arg1 = 0;
                message.arg2 = progress.getProgress();
                Bundle bundle = new Bundle();
                bundle.putInt(TRAIN_EPOCH, progress.getEpoch());
                bundle.putFloat(TRAIN_LOSS, progress.getLoss());
                bundle.putFloat(TRAIN_ACCURACY, progress.getAccuracy());
                message.setData(bundle);
                callbackMessage(msg.replyTo, message);
            } else if (MSG_BIND_EDGE == msg.what) {
                String bindId = msg.getData().getString(BIND_EDGE_ID);
                fedEdgeTrainApi.bindEdge(bindId);
            }
        }
    };
    private final Messenger mServiceMessenger = new Messenger(serviceHandler);

    @Override
    public void onCreate() {
        super.onCreate();
        fedEdgeTrainApi.init(getApplicationContext(), onTrainingStatusListener, onAccuracyLossListener);
        LogHelper.d("onCreate privatePath:%s", SharePreferencesData.getPrivatePath());
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Intent innerIntent = new Intent(this, GrayInnerService.class);
        startService(innerIntent);
        startForeground(GRAY_SERVICE_ID, new Notification());
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return mServiceMessenger.getBinder();
    }


    private void callbackMessage(@NonNull final Messenger clientMessenger, @NonNull final Message message) {
        try {
            clientMessenger.send(message);
        } catch (RemoteException e) {
            LogHelper.e(e, "callbackMessage RemoteException");
        }
    }

    private void sendMessageToClient(@NonNull final Message message) {
        if (mClientMessenger == null) {
            LogHelper.wtf("sendMessageToClient mClientMessenger is null.");
            return;
        }
        try {
            mClientMessenger.send(message);
        } catch (RemoteException e) {
            LogHelper.e(e, "sendMessageToClient RemoteException");
        }
    }

    public static class GrayInnerService extends Service {

        @Override
        public int onStartCommand(Intent intent, int flags, int startId) {
            startForeground(GRAY_SERVICE_ID, new Notification());
            stopForeground(true);
            stopSelf();
            return super.onStartCommand(intent, flags, startId);
        }

        @Nullable
        @Override
        public IBinder onBind(Intent intent) {
            return null;
        }
    }
}
