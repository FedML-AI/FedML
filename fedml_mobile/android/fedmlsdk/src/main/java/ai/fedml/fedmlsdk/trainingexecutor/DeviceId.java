package ai.fedml.fedmlsdk.trainingexecutor;

import android.content.Context;
import android.content.SharedPreferences;
import android.text.TextUtils;
import android.util.Log;

import androidx.ads.identifier.AdvertisingIdClient;
import androidx.ads.identifier.AdvertisingIdInfo;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.UUID;
import java.util.concurrent.Executors;

import ai.fedml.fedmlsdk.ContextHolder;

public class DeviceId {

    private static final String DEVICE_ID = "DeviceId";
    private String deviceId;

    DeviceId() {
        determineAdvertisingInfo();
    }

    public String getDeviceId() {
        if (!TextUtils.isEmpty(deviceId)) {
            return deviceId;
        }
        return getUuidAsDeviceId();
    }

    private void determineAdvertisingInfo() {
        final Context context = ContextHolder.getAppContext();
        if (AdvertisingIdClient.isAdvertisingIdProviderAvailable(context)) {
            ListenableFuture<AdvertisingIdInfo> advertisingIdInfoListenableFuture =
                    AdvertisingIdClient.getAdvertisingIdInfo(context);
            Futures.addCallback(advertisingIdInfoListenableFuture,
                    new FutureCallback<AdvertisingIdInfo>() {
                        @Override
                        public void onSuccess(AdvertisingIdInfo adInfo) {
                            deviceId = adInfo.getId();
                            String providerPackageName =
                                    adInfo.getProviderPackageName();
                            boolean isLimitTrackingEnabled =
                                    adInfo.isLimitAdTrackingEnabled();
                            Log.d("DeviceId", "adInfo id:" + deviceId + ",PackageName:"
                                    + providerPackageName + ",isLimit:" + isLimitTrackingEnabled);

                        }

                        @Override
                        public void onFailure(Throwable throwable) {
                            Log.e("DeviceId",
                                    "Failed to connect to Advertising ID provider.", throwable);
                            getUuidAsDeviceId();

                        }
                    }, Executors.newSingleThreadExecutor());
        } else {
            getUuidAsDeviceId();
        }
    }

    private String getUuidAsDeviceId() {
        final Context context = ContextHolder.getAppContext();
        SharedPreferences sp = context.getSharedPreferences(DEVICE_ID, Context.MODE_PRIVATE);
        String id = sp.getString(DEVICE_ID, "");
        if (TextUtils.isEmpty(id)) {
            final String uuid = UUID.randomUUID().toString();
            id = uuid;
            sp.edit().putString(DEVICE_ID, uuid).apply();
        }
        return id;
    }

}
