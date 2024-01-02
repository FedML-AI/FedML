package ai.fedml.edge.service.communicator;

import android.os.SystemClock;

import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallbackExtended;
import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.json.JSONException;
import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;

import ai.fedml.edge.constants.FedMqttTopic;
import ai.fedml.edge.request.response.ConfigResponse;
import ai.fedml.edge.service.Initializer;
import ai.fedml.edge.service.communicator.message.BaseMessage;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.utils.DeviceUtils;
import ai.fedml.edge.utils.GsonUtils;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;

import androidx.annotation.NonNull;

public class EdgeCommunicator implements MqttCallbackExtended {
    private static final int QOS = 2;
    private static final Properties CONN_PROPERTIES = new Properties() {
        {
            put("account_id", "ak");
        }
    };
    private final MemoryPersistence persistence = new MemoryPersistence();
    private final MqttConnectOptions connOpts;
    private final MqttClient client;
    private final Map<String, OnReceivedListener> subscribeTopics = new HashMap<>();

    private final List<OnReceivedListener> receivedListeners = new ArrayList<>();

    private final static class LazyHolder {
        private final static EdgeCommunicator sEdgeCommunicator = new EdgeCommunicator();
    }

    public static EdgeCommunicator getInstance() {
        return LazyHolder.sEdgeCommunicator;
    }

    public EdgeCommunicator() {
        final String deviceId = DeviceUtils.getDeviceId();
        ConfigResponse.MqttConfig mqttConfig = Initializer.getInstance().getMqttConfig();
        if (mqttConfig == null) {
            throw new EdgeCommunicatorException("fetch Mqtt Config failed!");
        }

        String serverURI = String.format(Locale.US, "tcp://%s:%d",
                mqttConfig.getHost(), mqttConfig.getPort());
        // EMQX Connect Options
        connOpts = new MqttConnectOptions();
        connOpts.setUserName(mqttConfig.getUser());
        connOpts.setPassword(mqttConfig.getPassword().toCharArray());
        connOpts.setCustomWebSocketHeaders(CONN_PROPERTIES);
        connOpts.setCleanSession(false);
        connOpts.setConnectionTimeout(30);
        connOpts.setKeepAliveInterval(mqttConfig.getKeepAlive());
        connOpts.setAutomaticReconnect(true);
        connOpts.setMaxInflight(10000);
        connOpts.setWill(FedMqttTopic.MQTT_LAST_WILL_TOPIC, buildLastWillPayload(),
                2, true);
        client = createMqttClient(serverURI, deviceId + "_" + System.currentTimeMillis());
    }

    private byte[] buildLastWillPayload() {
        String edgeId = SharePreferencesData.getBindingId();
        return ("{\"ID\":\"" + edgeId + "\",\"status\":\"" +
                MessageDefine.MSG_MLOPS_CLIENT_STATUS_OFFLINE + "\"}").getBytes();
    }

    private void connectMqtt() {
        try {
            LogHelper.i("EdgeCommunicator Connecting to broker:" + client.getServerURI());
            client.connect(connOpts);
        } catch (MqttException e) {
            LogHelper.e(e, "mqtt connect exception.");
            disconnect();
            SystemClock.sleep(5_000);
            connectMqtt();
        }
    }

    public void connect() {
        connectMqtt();
    }

    private MqttClient createMqttClient(@NonNull final String serverURI, @NonNull final String clientId) {
        MqttClient mqttClient;
        try {
            mqttClient = new MqttClient(serverURI, clientId, persistence);
            mqttClient.setCallback(this);
            mqttClient.setTimeToWait(5000);
        } catch (MqttException e) {
            throw new EdgeCommunicatorException("mqtt client create failed", e);
        }
        return mqttClient;
    }

    @Override
    public void connectComplete(boolean reconnect, String serverURI) {
        LogHelper.i("FedMLDebug. EdgeCommunicator Connected reconnect:" + reconnect + ", uri:" + serverURI + ", subscribeTopics: " + subscribeTopics);
        try {
            // subscribe the default topic
            client.subscribe("edge/" + DeviceUtils.getDeviceId());
            // subscribe the topics
            if (!subscribeTopics.isEmpty()) {
                client.subscribe(subscribeTopics.keySet().toArray(new String[0]));
                LogHelper.i("EdgeCommunicator Connected subscribe:" + subscribeTopics.keySet());
                for (Map.Entry<String, OnReceivedListener> entry : subscribeTopics.entrySet()) {
                    if (entry.getKey().startsWith("fedml_")) {
                        notifyConnectionReady(entry.getKey(), entry.getValue());
                    }
                }
            }
            if (!receivedListeners.isEmpty()) {
                for (OnReceivedListener listener : receivedListeners) {
                    notifyConnectionReady("connectionReady", listener);
                }
            }
        } catch (MqttException e) {
            LogHelper.e(e, "mqtt subscribe exception.");
        }
    }

    @Override
    public void connectionLost(Throwable cause) {
        LogHelper.w(cause, "connection Lost can re-connect!");
    }

    @Override
    public void messageArrived(@NonNull String topic, @NonNull MqttMessage message) {
        LogHelper.i("FedMLDebug. EdgeCommunicator messageArrived topic:%s, Qos:%d, Content:%s", topic, message.getQos(),
                new String(message.getPayload()));
        OnReceivedListener receivedListener = subscribeTopics.get(topic);
        if (receivedListener != null) {
            if (message.isRetained())
                return;
            receivedListener.onReceived(topic, message.getPayload());
        }
    }

    @Override
    public void deliveryComplete(IMqttDeliveryToken token) {
        //LogHelper.d("EdgeCommunicator deliveryComplete---------" + token.isComplete());
    }

    public void subscribe(@NonNull String topic, @NonNull OnReceivedListener listener) {
        // TODO: try three times
        subscribeTopics.put(topic, listener);
        LogHelper.i("FedMLDebug. EdgeCommunicator subscribe topic:%s", topic);
        if (client != null && client.isConnected()) {
            try {
                client.subscribe(topic, QOS);
            } catch (MqttException e) {
                LogHelper.e(e, "subscribe exception.");
            }
        } else {
            LogHelper.i("subscribe delay to connected");
        }
    }

    public void addListener(@NonNull OnReceivedListener listener) {
        receivedListeners.add(listener);
    }

    public void unsubscribe(@NonNull final String topic) {
        LogHelper.i("EdgeCommunicator unsubscribe topic:%s", topic);
        subscribeTopics.remove(topic);
        if (client != null && client.isConnected()) {
            try {
                client.unsubscribe(topic);
            } catch (MqttException e) {
                LogHelper.e(e, "unsubscribe exception.");
            }
        }
    }

    public boolean sendMessage(@NonNull String topic, @NonNull String msg) {
        // TODO: try three times
        if (client == null) {
            LogHelper.e("mqtt client is not initial, when sendMessage(%s, %s)", topic, msg);
            return false;
        }
        MqttMessage message = new MqttMessage(msg.getBytes());
        message.setQos(QOS);
        message.setRetained(true);
        try {
            client.publish(topic, message);
            LogHelper.i("FedMLDebug. sendMessage(%s)", topic);
            return true;
        } catch (MqttException e) {
            LogHelper.e(e, "FedMLDebug. Mqtt publish failed！(%s, %s)", topic, msg);
        }
        return false;
    }

    public void disconnect() {
        if (client == null) {
            LogHelper.e("mqtt client is not initial, when disconnect");
            return;
        }
        try {
            client.disconnect();
            LogHelper.i("MQTT Disconnected");
        } catch (MqttException e) {
            LogHelper.e(e, "disconnect exception.");
        }
    }

    public boolean sendMessage(final String topic, @NonNull BaseMessage msg) {
        return sendMessage(topic, GsonUtils.toJson(msg));
    }

    private void notifyConnectionReady(final String topic, final OnReceivedListener receivedListener) {
        if (receivedListener instanceof OnMqttConnectionReadyListener) {
            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put(MessageDefine.MSG_TYPE, MessageDefine.MSG_TYPE_CONNECTION_IS_READY);
            } catch (JSONException e) {
                LogHelper.e(e, "notifyConnectionReady JSON put failed.");
            }
            receivedListener.onReceived(topic, jsonObject.toString().getBytes(StandardCharsets.UTF_8));
        }
        LogHelper.i("FedMLDebug. notifyConnectionReady");
    }
}
