package ai.febml.febmlmobile.trainingexecutor;

import android.util.Log;

import androidx.annotation.NonNull;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import ai.febml.febmlmobile.utils.StorageUtils;
import lombok.SneakyThrows;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;

public class TrainingExecutor {
    private static final String TAG = "TrainingExecutor";
    private static final int QOS = 2;
    private final ITrainingExecutorService executorService;
    private final Gson gson;
    private final DeviceId deviceId;
    private final MemoryPersistence persistence = new MemoryPersistence();
    private final String broker;
    private MqttClient client;
    private String executorTopic;
    private final MqttConnectOptions connOpts;

    public TrainingExecutor(@NonNull final String baseUrl, @NonNull final String broker) {
        Retrofit retrofit = new Retrofit.Builder().baseUrl(baseUrl).build();
        executorService = retrofit.create(ITrainingExecutorService.class);
        gson = new GsonBuilder().create();
        deviceId = new DeviceId();
        this.broker = broker;
        connOpts = new MqttConnectOptions();
        connOpts.setUserName("emqx_test");
        connOpts.setPassword("emqx_test_password".toCharArray());
        // 保留会话
        connOpts.setCleanSession(true);
    }

    public void init() {
        onLine(DeviceInfo.builder().deviceId(deviceId.getDeviceId()).build());
        initMqttClient();
    }

    public void onLine(DeviceInfo deviceInfo) {
        final String json = gson.toJson(deviceInfo);
        Log.d(TAG, "onLine: " + json);
        MediaType mediaType = MediaType.parse("application/json");
        RequestBody req = RequestBody.create(json, mediaType);
        Call<ResponseBody> call = executorService.deviceOnLine(req);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(@NonNull Call<ResponseBody> call, @NonNull Response<ResponseBody> response) {
                DeviceOnLineResponse onLineResponse = gson.fromJson(new InputStreamReader(response.body().byteStream()),
                        DeviceOnLineResponse.class);
                Log.d(TAG, "onResponse: " + onLineResponse);
                executorTopic = onLineResponse.getExecutorTopic();
            }

            @Override
            public void onFailure(@NonNull Call<ResponseBody> call, @NonNull Throwable t) {
                Log.w(TAG, "onFailure: ", t);
            }
        });
    }

    public void uploadFile(@NonNull final String fileName, @NonNull final String filePath) {
        RequestBody nameReq = RequestBody.create(fileName, MediaType.parse("application/json;charset=UTF-8"));
        RequestBody requestFile = RequestBody.create(new File(filePath), MediaType.parse("application/octet-stream"));
        MultipartBody.Part filePart = MultipartBody.Part.createFormData("model_file", fileName, requestFile);
        Call<ResponseBody> call = executorService.upload(nameReq, filePart);
        call.enqueue(new Callback<ResponseBody>() {
            @SneakyThrows
            @Override
            public void onResponse(@NonNull Call<ResponseBody> call, @NonNull Response<ResponseBody> response) {
                Log.d(TAG, "onResponse: " + response.body().string());
            }

            @Override
            public void onFailure(@NonNull Call<ResponseBody> call, @NonNull Throwable t) {
                Log.w(TAG, "onFailure: ", t);
            }
        });
    }

    public void downloadFile(@NonNull final String fileName, @NonNull final String url) {
        Call<ResponseBody> call = executorService.downloadFile(url);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(@NonNull Call<ResponseBody> call, @NonNull Response<ResponseBody> response) {
                Log.d(TAG, "onResponse: " + response.isSuccessful());
                assert response.body() != null;
                StorageUtils.saveToLabelDataPath(response.body().byteStream(), fileName);
            }

            @Override
            public void onFailure(@NonNull Call<ResponseBody> call, @NonNull Throwable t) {
                Log.w(TAG, "onFailure: ", t);
            }
        });
    }

    private void initMqttClient() {
        try {
            client = new MqttClient(broker, deviceId.getDeviceId(), persistence);
            // 设置回调
            client.setCallback(new OnMessageCallback());
            // 建立连接
            System.out.println("Connecting to broker: " + broker);
            client.connect(connOpts);
            client.subscribe("device");
            System.out.println("Connected");
        } catch (MqttException e) {
            Log.w(TAG, "mqtt create: ", e);
        }

    }

    public void destroy() {
        try {
            client.disconnect();
            System.out.println("Disconnected");
            client.close();
        } catch (MqttException e) {
            e.printStackTrace();
        }

    }

    public void sendMessage(@NonNull String msg) {
        // 消息发布所需参数
        MqttMessage message = new MqttMessage(msg.getBytes());
        message.setQos(QOS);
        try {
            client.publish(executorTopic, message);
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }

    public class OnMessageCallback implements MqttCallback {
        public void connectionLost(Throwable cause) {
            // 连接丢失后，一般在这里面进行重连
            System.out.println("连接断开，可以做重连");
            try {
                client.connect(connOpts);
                client.subscribe("device");
            } catch (MqttException e) {
                e.printStackTrace();
            }
            System.out.println("Connected");
        }

        public void messageArrived(String topic, MqttMessage message) throws Exception {
            // subscribe后得到的消息会执行到这里面
            System.out.println("接收消息主题:" + topic);
            System.out.println("接收消息Qos:" + message.getQos());
            System.out.println("接收消息内容:" + new String(message.getPayload()));
        }

        public void deliveryComplete(IMqttDeliveryToken token) {
            System.out.println("deliveryComplete---------" + token.isComplete());
        }
    }
}
