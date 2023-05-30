package ai.fedml.edge.service.component;

import android.os.Handler;

import org.json.JSONException;
import org.json.JSONObject;

import ai.fedml.edge.constants.FedMqttTopic;
import ai.fedml.edge.service.ContextHolder;
import ai.fedml.edge.service.communicator.EdgeCommunicator;
import ai.fedml.edge.utils.BackgroundHandler;
import ai.fedml.edge.utils.BatteryUtils;
import ai.fedml.edge.utils.DeviceUtils;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.MemoryUtils;
import ai.fedml.edge.utils.entity.Battery;
import ai.fedml.edge.utils.entity.Memory;

public class DeviceInfoReporter {

    private final Handler mBgHandler;

    private final long mEdgeId;

    private final Runnable mRunnable;

    private final EdgeCommunicator mEdgeCommunicator;

    public DeviceInfoReporter(final long edgeId, EdgeCommunicator edgeCommunicator) {
        mEdgeId = edgeId;
        mEdgeCommunicator = edgeCommunicator;
        mBgHandler = new BackgroundHandler("DeviceInfoReporter");
        mRunnable = new Runnable() {
            @Override
            public void run() {
                sendDeviceInfo();
                mBgHandler.postDelayed(this, 10000L);
            }
        };
    }

    public void start() {
        mBgHandler.postDelayed(mRunnable, 10000L);
    }

    public void release() {
        mBgHandler.removeCallbacksAndMessages(null);
    }

    private void sendDeviceInfo() {
        Battery battery = BatteryUtils.getBattery(ContextHolder.getAppContext());
        Memory memory = MemoryUtils.getMemory(ContextHolder.getAppContext());
        final SysStats sysStats = SysStats.getInstance();
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("edgeId", mEdgeId);
            jsonObject.put("networkType", DeviceUtils.getNetworkType(ContextHolder.getAppContext()));
            jsonObject.put("batteryStatus", battery.getStatus());
            jsonObject.put("batteryPower", battery.getPower());
            jsonObject.put("batteryPercent", battery.getPercentage());
            jsonObject.put("batteryHealth", battery.getHealth());
            jsonObject.put("ramMemoryTotal", memory.getRamMemoryTotal());
            jsonObject.put("ramMemoryAvailable", memory.getRamMemoryAvailable());
            jsonObject.put("romMemoryAvailable", memory.getRomMemoryAvailable());
            jsonObject.put("romMemoryTotal", memory.getRomMemoryTotal());
            Float cpuUtilization = sysStats.getCpuUtilization();
            if (null != cpuUtilization) {
                jsonObject.put("cpuUtilization", String.valueOf(cpuUtilization));
            }
        } catch (JSONException e) {
            LogHelper.e(e, "sendDeviceInfo(%s)", mEdgeId);
        }
        mEdgeCommunicator.sendMessage(FedMqttTopic.DEVICE_INFO, jsonObject.toString());
    }
}
