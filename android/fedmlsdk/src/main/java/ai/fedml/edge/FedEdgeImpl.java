package ai.fedml.edge;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;
import android.text.TextUtils;

import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.request.listener.OnBindingListener;
import ai.fedml.edge.request.listener.OnUnboundListener;
import ai.fedml.edge.request.listener.OnUserInfoListener;
import ai.fedml.edge.request.parameter.BindingAccountReq;
import ai.fedml.edge.service.ContextHolder;
import ai.fedml.edge.service.EdgeService;
import ai.fedml.edge.utils.AesUtil;
import ai.fedml.edge.utils.CpuUtils;
import ai.fedml.edge.utils.DeviceUtils;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.MemoryUtils;
import ai.fedml.edge.utils.ObfuscatedString;
import ai.fedml.edge.utils.preference.SharePreferencesData;
import androidx.annotation.NonNull;

class FedEdgeImpl implements EdgeMessageDefine, FedEdgeApi {

    /**
     * mainfest meata key "fedml_key"
     */
    private static final String META_ACCOUNT_KEY = new ObfuscatedString(new long[]{0x78DA743E5BE2970DL,
            0x380F3AEE359ADEEEL, 0x77E0C41263DBC235L}).toString();
    /**
     * SecretKey: ks-FedML-beehive
     */
    private static final String SECRET_KEY = new ObfuscatedString(new long[]{0xBA683391111A600DL, 0x84924D54717A16E1L,
            0xBE985554215915ACL}).toString();
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
            LogHelper.d("FedEdge receive message from service:%s", msg.toString());
            if (MSG_TRAIN_STATUS == msg.what) {
                LogHelper.d("FedEdge MSG_TRAIN_STATUS. msg.arg1:%d", msg.arg1);
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
    private volatile boolean canInit = false;


    public void init(Context appContext) {
        canInit = true;
        ContextHolder.initialize(appContext);
        final String processName = DeviceUtils.getProcessName();
        LogHelper.i("FedEdge init, processName:%s", processName);
        if (!TextUtils.isEmpty(processName) && appContext.getPackageName().equals(processName)) {
            bindService();
            initBindingState(appContext);
        }
    }

    @Override
    public void bindingAccount(@NonNull String accountId, @NonNull String deviceId, @NonNull OnBindingListener listener) {
        BindingAccountReq req = BindingAccountReq.builder()
                .accountId(accountId).deviceId(deviceId)
                .cpuAbi(CpuUtils.getInstance().getCpuAbi())
                .osVersion(Build.VERSION.RELEASE).memory(MemoryUtils.getMemory(ContextHolder.getAppContext()).getRamMemoryTotal())
                .build();
        RequestManager.bindingAccount(req, listener);
    }

    @Override
    public void unboundAccount(@NonNull String edgeId, @NonNull OnUnboundListener listener) {
        RequestManager.unboundAccount(edgeId, listener);
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

    @Override
    public void getUserInfo(@NonNull OnUserInfoListener listener) {
        RequestManager.getUserInfo(listener);
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

    @Override
    public void unInit() {
        final String processName = DeviceUtils.getProcessName();
        LogHelper.i("FedEdge unInit, processName:%s", processName);
        Context appContext = ContextHolder.getAppContext();
        // Only the main process can unInit
        if (!TextUtils.isEmpty(processName) && appContext.getPackageName().equals(processName)) {
            canInit = false;
            sendMessage(MSG_STOP_EDGE_SERVICE, null);
            // if a stopped service still has ServiceConnection objects bound to it with the BIND_AUTO_CREATE set,
            // it will not be destroyed until all of these bindings are removed.
            appContext.unbindService(mServiceConnection);
            // EdgeService will call onDestroy()
            Intent intent = new Intent(appContext, EdgeService.class);
            appContext.stopService(intent);
        }
    }

    private void bindService() {
        if (!canInit){
            LogHelper.w("The service is already uninit and cannot start edge Service!");
        }
        Context appContext = ContextHolder.getAppContext();
        Intent intent = new Intent(appContext, EdgeService.class);
//        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
//            // The foreground service can only be started through startForegroundService above android8.0.
//            appContext.startForegroundService (intent);
//        }
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
            LogHelper.e(e, "FedEdge sendMessage failed!");
        }
    }

    private String getAccountFromMeta(@NonNull final Context context) {
        try {
            ApplicationInfo appInfo = context.getPackageManager()
                    .getApplicationInfo(context.getPackageName(), PackageManager.GET_META_DATA);
            String cipherAccountId = appInfo.metaData.getString(META_ACCOUNT_KEY);
            String accountIdString = AesUtil.decrypt(cipherAccountId, SECRET_KEY);
            LogHelper.d("accountId=%s", accountIdString);
            return accountIdString;
        } catch (PackageManager.NameNotFoundException e) {
            LogHelper.e(e, "metaData get failed.");
        }
        return null;
    }

    private void initBindingState(@NonNull final Context context) {
        final String bindingId = SharePreferencesData.getBindingId();
        LogHelper.d("initBindingState bindingId: %s", bindingId);
        // TODO: Whether there is no need to rebind if it is already bound？
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
                .accountId(accountId).deviceId(deviceId)
                .cpuAbi(CpuUtils.getInstance().getCpuAbi())
                .osVersion(Build.VERSION.RELEASE).memory(MemoryUtils.getMemory(ContextHolder.getAppContext()).getRamMemoryTotal())
                .build();
        RequestManager.bindingAccount(req, data -> {
            LogHelper.d("initBindingState bindingData.getBindingId() = %s", data);
            if (data != null) {
                bindEdge(data.getBindingId());
            }
        });
    }
}
