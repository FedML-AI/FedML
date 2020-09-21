package ai.fedml.iot.mq;

import ai.fedml.iot.Config;
import ai.fedml.iot.device.IoTDevice;
import ai.fedml.iot.utils.LogUtils;

import org.fusesource.hawtbuf.Buffer;
import org.fusesource.hawtbuf.UTF8Buffer;
import org.fusesource.mqtt.client.Callback;
import org.fusesource.mqtt.client.CallbackConnection;
import org.fusesource.mqtt.client.FutureConnection;
import org.fusesource.mqtt.client.Listener;
import org.fusesource.mqtt.client.MQTT;
import org.fusesource.mqtt.client.QoS;

import java.net.URISyntaxException;

public class MQManager {
    private static final String TAG = MQManager.class.getSimpleName();
    private final static String CONNECTION_URL = Config.BASE_URL_IOT_MQ_SERVICE;

    private final static boolean CLEAN_START = true;
    private final static short KEEP_ALIVE = 30;

    private static String mPublishTopic = "FedML/mobile";

    public final static long RECONNECTION_ATTEMPT_MAX = 4;
    public final static long RECONNECTION_DELAY = 2000;

    public final static int SEND_BUFFER_SIZE = 64 * 1024;//maximum cache

    private FutureConnection connection = null;
    private MQTT mqtt = null;

    public static final int CONNECT_STATE_DISCONNECTED = 1;
    public static final int CONNECT_STATE_CONNECTING = 2;
    public static final int CONNECT_STATE_CONNECTED = 3;

    private int mConnectState = CONNECT_STATE_DISCONNECTED;

    public synchronized void connect(final IMQListener listener) throws Exception {
        if(mConnectState == CONNECT_STATE_CONNECTING || mConnectState == CONNECT_STATE_CONNECTED){
            return;
        }
        mConnectState = CONNECT_STATE_CONNECTING;
        mqtt = new MQTT();
        try {
            mqtt.setHost(CONNECTION_URL);
            mqtt.setCleanSession(CLEAN_START);
            mqtt.setReconnectAttemptsMax(RECONNECTION_ATTEMPT_MAX);
            mqtt.setReconnectDelay(RECONNECTION_DELAY);
            mqtt.setKeepAlive(KEEP_ALIVE);
            mqtt.setSendBufferSize(SEND_BUFFER_SIZE);
            mqtt.setUserName(Config.MQ_USER);
            mqtt.setPassword(Config.MQ_PASSWORD);
            mqtt.setVersion("3.1.1");
            LogUtils.e(TAG, "mqtt version = " + mqtt.getVersion());

            String strClientID = IoTDevice.getDeviceID();
            LogUtils.e(TAG, "strClientID = " + strClientID);
            mqtt.setClientId(strClientID);

            final CallbackConnection callbackConnection = mqtt.callbackConnection();
            callbackConnection.listener(new Listener() {
                @Override
                public void onConnected() {

                }

                @Override
                public void onDisconnected() {
                }

                @Override

                public void onPublish(UTF8Buffer utf8Buffer, Buffer buffer, Runnable runnable) {
                    if (listener != null) {
                        listener.onPublish(utf8Buffer, buffer, runnable);
                    }
                    runnable.run();
                }

                @Override
                public void onFailure(Throwable throwable) {
                }
            });


            callbackConnection.connect(new Callback<Void>() {
                @Override
                public void onSuccess(Void aVoid) {
                    mConnectState = CONNECT_STATE_CONNECTED;
                    if (listener != null) {
                        listener.onConnected();
                    }

                    LogUtils.e(TAG, "MQTT connect successfully!");
                }

                @Override
                public void onFailure(Throwable throwable) {
                    if (listener != null) {
                        listener.onFailure(throwable);
                    }
                    mConnectState = CONNECT_STATE_DISCONNECTED;
                    LogUtils.d(TAG, "MQTT subscribe failed!");
                    if(callbackConnection != null){
                        callbackConnection.disconnect(null);
                    }
                }
            });

            connection = mqtt.futureConnection();
            connection.connect();
        } catch (URISyntaxException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void release() {
        if (mqtt == null || connection == null) return;
        mqtt = null;
        connection.disconnect();
        connection.kill();
        connection = null;
        mConnectState = CONNECT_STATE_DISCONNECTED;
    }

    public int getState(){
        return mConnectState;
    }

    public boolean sendLocation(FLMessage flmsg) {
        if (mConnectState == CONNECT_STATE_DISCONNECTED) {
            LogUtils.d(TAG, "MQ disconnected");
            return false;
        }
        byte[] message = flmsg.toByteArray();
        //LogUtils.e(TAG, "sendlocation md5 = " + DigestUtils.md5Hex(locationMessage.toString()));
        connection.publish(mPublishTopic, message, QoS.AT_LEAST_ONCE,
                false);

        //if(Config.bDebugMode){
            LogUtils.e(TAG, "MQTT has sent a Message." +
                    " size = " + message.length + " ." + " Payload: " + "\n" + flmsg.toString());
        //}
        return true;
    }
}