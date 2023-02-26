package ai.fedml.edge;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;
import android.text.TextUtils;
import android.util.Log;

import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.request.parameter.BindingAccountReq;
import ai.fedml.edge.service.ContextHolder;
import ai.fedml.edge.service.EdgeService;
import ai.fedml.edge.utils.DeviceUtils;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;
import androidx.annotation.NonNull;

class FedEdgeImpl implements EdgeMessageDefine, FedEdgeApi {
    private static final String TAG = "FedEdgeManager";
    private final ServiceConnection mServiceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            mServiceMessenger = new Messenger(service);
            Bundle bundle = new Bundle();
            bundle.putString("msg", "how are you?");
            sendMessage(MSG_FROM_CLIENT, bundle);
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            mServiceMessenger = null;
            bindService();
        }
    };
    private final Handler mClientHandler = new Handler(Looper.getMainLooper()) {
        @Override
        public void handleMessage(Message msg) {
            Log.d(TAG, "receive message from service:" + msg.toString());
            if (MSG_TRAIN_STATUS == msg.what) {
                if (onTrainingStatusListener != null) {
                    onTrainingStatusListener.onStatusChanged(msg.arg1);
                }
            } else if (MSG_TRAIN_PROGRESS == msg.what) {
                if (onTrainProgressListener != null) {
                    onTrainProgressListener.onProgressChanged(msg.arg1, msg.arg2);
                }
            } else if (MSG_TRAIN_LOSS == msg.what) {
                Bundle bundle = msg.getData();
                if (onTrainProgressListener != null && bundle != null) {
                    onTrainProgressListener.onEpochLoss(msg.arg1, msg.arg2,
                            bundle.getFloat(TRAIN_LOSS, 0));
                }
            } else if (MSG_TRAIN_ACCURACY == msg.what) {
                Bundle bundle = msg.getData();
                if (onTrainProgressListener != null && bundle != null) {
                    onTrainProgressListener.onEpochAccuracy(msg.arg1, msg.arg2,
                            bundle.getFloat(TRAIN_ACCURACY, 0));
                }
            }
        }
    };
    private final Messenger mClientMessenger = new Messenger(mClientHandler);
    private Messenger mServiceMessenger;
    private OnTrainingStatusListener onTrainingStatusListener;
    private OnTrainProgressListener onTrainProgressListener;


    public void init(Context appContext) {
        ContextHolder.initialize(appContext);
        final String processName = DeviceUtils.getProcessName();
        Log.i(TAG, "init " + processName);
        if (!TextUtils.isEmpty(processName) && appContext.getPackageName().equals(processName)) {
            bindService();
            initBindingState(appContext);
        }
    }

    @Override
    public String getBoundEdgeId() {
        return SharePreferencesData.getBindingId();
    }

    @Override
    public void bindEdge(String bindId) {
        Bundle bundle = new Bundle();
        bundle.putString(BIND_EDGE_ID, bindId);
        sendMessage(MSG_BIND_EDGE, bundle);
    }

    public void train() {
        Bundle bundle = new Bundle();
        bundle.putString(TRAIN_ARGS, "how are you?");
        sendMessage(MSG_START_TRAIN, bundle);
    }

    @Override
    public void getTrainingStatus() {
        sendMessage(MSG_TRAIN_STATUS, new Bundle());
    }

    @Override
    public void getEpochAndLoss() {
        sendMessage(MSG_TRAIN_PROGRESS, new Bundle());
    }

    @Override
    public void setTrainingStatusListener(OnTrainingStatusListener listener) {
        onTrainingStatusListener = listener;
    }

    @Override
    public void setEpochLossListener(OnTrainProgressListener listener) {
        onTrainProgressListener = listener;
    }

    @Override
    public String getHyperParameters() {
        return SharePreferencesData.getHyperParameters();
    }

    @Override
    public void setPrivatePath(final String path) {
        SharePreferencesData.savePrivatePath(path);
    }

    @Override
    public String getPrivatePath() {
        return SharePreferencesData.getPrivatePath();
    }

    private void bindService() {
        Context appContext = ContextHolder.getAppContext();
        Intent intent = new Intent(appContext, EdgeService.class);
        appContext.bindService(intent, mServiceConnection, Context.BIND_AUTO_CREATE);
    }

    private void sendMessage(final int action, final Bundle bundle) {
        Message message = Message.obtain();
        message.what = action;
        message.setData(bundle);
        message.replyTo = mClientMessenger;
        try {
            mServiceMessenger.send(message);
        } catch (RemoteException e) {
            Log.e(TAG, "sendMessage failed!", e);
        }
    }

    private String getAccountFromMeta(@NonNull final Context context) {
        try {
            ApplicationInfo appInfo = context.getPackageManager()
                    .getApplicationInfo(context.getPackageName(), PackageManager.GET_META_DATA);
            int accountId = appInfo.metaData.getInt("fedml_account", 0);
            if (accountId != 0) {
                return String.valueOf(accountId);
            }
            return appInfo.metaData.getString("fedml_account");
        } catch (PackageManager.NameNotFoundException e) {
            LogHelper.e(e, "metaData get failed.");
        }
        return null;
    }

    private void initBindingState(@NonNull final Context context) {
        final String bindingId = SharePreferencesData.getBindingId();
        LogHelper.d("initBindingState bindingId: %s", bindingId);
        String accountId = getAccountFromMeta(context);
        final String deviceId = DeviceUtils.getDeviceId();
        LogHelper.d("initBindingState AccountFromMeta: %s, deviceId: %s", accountId, deviceId);
        if (TextUtils.isEmpty(accountId)) {
            accountId = SharePreferencesData.getAccountId();
        }
        if (TextUtils.isEmpty(accountId)) {
            return;
        }
        BindingAccountReq req = BindingAccountReq.builder()
                .accountId(accountId).deviceId(deviceId).build();
        RequestManager.bindingAccount(req, data -> {
            LogHelper.d("initBindingState bindingData.getBindingId() = %s", data);
            if (data != null) {
                bindEdge(data.getBindingId());
            }
        });
    }
}
