package ai.fedml.edge.service;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.PowerManager;
import android.os.RemoteException;

import ai.fedml.edge.EdgeMessageDefine;
import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.R;
import ai.fedml.edge.service.entity.TrainProgress;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;
import androidx.annotation.NonNull;

public class EdgeService extends Service implements EdgeMessageDefine {
    private static final int GRAY_SERVICE_ID = 1001;
    private static final String NOTIFY_CHANNEL_ID = "FedML-Edge";
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
            LogHelper.d("FedMLDebug. round = %d, epoch = %d, accuracy = %f", round, epoch, accuracy);
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
                LogHelper.d("FedMLDebug. bindId = " + bindId);
                fedEdgeTrainApi.bindEdge(bindId);
            } else if (MSG_STOP_EDGE_SERVICE == msg.what) {
                LogHelper.d("FedMLDebug. STOP_EDGE_SERVICE");
                android.os.Process.killProcess(android.os.Process.myPid());
            }
        }
    };
    private final Messenger mServiceMessenger = new Messenger(serviceHandler);
    private PowerManager.WakeLock mWakeLock;
    private MediaPlayer mMediaPlayer;

    @Override
    public void onCreate() {
        super.onCreate();
        // Add power control and use the PowerManager.WakeLock object to keep CPU running.
        PowerManager pm = (PowerManager) getSystemService (POWER_SERVICE);
        mWakeLock = pm.newWakeLock (PowerManager.PARTIAL_WAKE_LOCK, EdgeService.class.getName ());
        mWakeLock.acquire ();
        // Play silent music to prevent resources from being released
        mMediaPlayer = MediaPlayer.create(getApplicationContext(), R.raw.no_kill);
        mMediaPlayer.setLooping(true);
        // Train init
        fedEdgeTrainApi.init(getApplicationContext(), onTrainingStatusListener, onAccuracyLossListener);
        LogHelper.d("onCreate privatePath:%s", SharePreferencesData.getPrivatePath());
        LogHelper.d("FedMLDebug. EdgeService onCreate()");
    }

    @Override
    public void onDestroy() {
        exitService();
        super.onDestroy();
        LogHelper.d("FedMLDebug. EdgeService onDestroy()");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        startForeground(GRAY_SERVICE_ID, buildNotification());
        if (mMediaPlayer != null) {
            mMediaPlayer.start();
        }
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
            LogHelper.w("sendMessageToClient mClientMessenger is null.");
            return;
        }
        try {
            mClientMessenger.send(message);
        } catch (RemoteException e) {
            LogHelper.e(e, "sendMessageToClient RemoteException");
        }
    }

    public void exitService() {
        stopForeground(true);
        if (mMediaPlayer != null) {
            mMediaPlayer.stop();
        }
        if (mWakeLock != null) {
            mWakeLock.release();
            mWakeLock = null;
        }
        this.stopSelf();
    }

    private static final String NOTIFICATION_CHANNEL_NAME = "LBSbackgroundLocation";
    private NotificationManager notificationManager = null;
    boolean isCreateChannel = false;
    @SuppressLint("NewApi")
    private Notification buildNotification() {
        Context appContext = ContextHolder.getAppContext();
        Notification.Builder builder = null;
        Notification notification = null;
        if(android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // Android O Notification need NotificationChannel
            if (null == notificationManager) {
                notificationManager = (NotificationManager) appContext.getSystemService(Context.NOTIFICATION_SERVICE);
            }
            final String channelId =  NOTIFY_CHANNEL_ID;
            if(!isCreateChannel) {
                NotificationChannel notificationChannel = new NotificationChannel(channelId,
                        NOTIFICATION_CHANNEL_NAME, NotificationManager.IMPORTANCE_DEFAULT);
                // Whether to display small dots in the upper right corner of the desktop icon
                notificationChannel.enableLights(false);
                // dot color
                notificationChannel.setLightColor(Color.TRANSPARENT);
                // whether to display the notification of this channel when you long press the desktop icon
                notificationChannel.setShowBadge(false);
                notificationManager.createNotificationChannel(notificationChannel);
                isCreateChannel = true;
            }
            builder = new Notification.Builder( appContext.getApplicationContext(), channelId);
        } else {
            builder = new Notification.Builder( appContext.getApplicationContext());
        }
        builder.setSmallIcon(R.mipmap.ic_logo)
                .setContentTitle(appContext.getPackageName())
                .setContentText("Running in background")
                .setContentIntent(PendingIntent.getActivity(this, 0,
                        this.getPackageManager().getLaunchIntentForPackage(this.getPackageName()),
                        PendingIntent.FLAG_UPDATE_CURRENT))
                .setWhen(System.currentTimeMillis());
        return builder.build();
    }
}
