package ai.fedml.edge.service.component;

import android.os.Handler;

import org.json.JSONException;
import org.json.JSONObject;

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

    private EdgeCommunicator mEdgeCommunicator;

    public DeviceInfoReporter(final long edgeId, EdgeCommunicator edgeCommunicator) {
        mEdgeId = edgeId;
        mEdgeCommunicator = edgeCommunicator;
        mBgHandler = new BackgroundHandler("DeviceInfoReporter");
        mRunnable = new Runnable() {
            @Override
            public void run() {
                sendDeviceInfo();
                mBgHandler.postDelayed(this, 3000L);
            }
        };
    }

    public void start() {
        mBgHandler.postDelayed(mRunnable, 3000L);
    }

    public void release() {
        mBgHandler.removeCallbacksAndMessages(null);
    }

    private void sendDeviceInfo() {
        final String topicMetrics = "fl_client/mlops/device_info";
        Battery battery = BatteryUtils.getBattery(ContextHolder.getAppContext());
        Memory memory = MemoryUtils.getMemory(ContextHolder.getAppContext());
        final SysStats sysStats = SysStats.getInstance();
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("edge_id", mEdgeId);
            jsonObject.put("network_type", DeviceUtils.getNetworkType(ContextHolder.getAppContext()));
            jsonObject.put("battery_status", battery.getStatus());
            jsonObject.put("battery_power", battery.getPower());
            jsonObject.put("battery_percent", battery.getPercentage());
            jsonObject.put("battery_health", battery.getHealth());
            jsonObject.put("ramMemoryTotal", memory.getRamMemoryTotal());
            jsonObject.put("ramMemoryAvailable", memory.getRamMemoryAvailable());
            jsonObject.put("romMemoryAvailable", memory.getRomMemoryAvailable());
            jsonObject.put("romMemoryTotal", memory.getRomMemoryTotal());
            jsonObject.put("cpu_utilization", sysStats.getCpuUtilization());
        } catch (JSONException e) {
            LogHelper.e(e, "sendDeviceInfo(%s)", mEdgeId);
        }
        mEdgeCommunicator.sendMessage(topicMetrics, jsonObject.toString());
    }
}
