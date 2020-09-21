package ai.fedml.fedmlsdk.trainingexecutor;

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
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.InputStreamReader;

import ai.fedml.fedmlsdk.FedMlTaskListener;
import ai.fedml.fedmlsdk.utils.StorageUtils;
import lombok.SneakyThrows;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class TrainingExecutor {
    private static final String TAG = "TrainingExecutor";
    private static final int QOS = 2;
    private final ITrainingExecutorService executorService;
    private final Gson gson;
    private final DeviceId deviceId;
    private final MemoryPersistence persistence = new MemoryPersistence();
    private final String broker;
    private MqttClient client;
    private String executorId;
    private String executorTopic = "executorTopic";
    private ExecutorResponse.TrainingTaskParam trainingTaskParam;
    private final MqttConnectOptions connOpts;
    private FedMlTaskListener mFedMlTaskListener;

    public TrainingExecutor(@NonNull final String baseUrl, @NonNull final String broker) {
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(baseUrl)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        executorService = retrofit.create(ITrainingExecutorService.class);
        gson = new GsonBuilder().create();
        deviceId = new DeviceId();
        this.broker = broker;
        connOpts = new MqttConnectOptions();
        connOpts.setUserName("emqx_fedml_mobile");
        connOpts.setPassword("fedml_password".toCharArray());
        // 保留会话
        connOpts.setCleanSession(true);
    }

    public void init() {
        registerDevice(DeviceInfo.builder().deviceId(deviceId.getDeviceId()).build());
        initMqttClient();
    }

    public void setFedMlTaskListener(FedMlTaskListener listener) {
        mFedMlTaskListener = listener;
    }

    public void registerDevice(@NonNull DeviceInfo deviceInfo) {
        Call<ExecutorResponse> call = executorService.registerDevice(deviceInfo.getDeviceId());
        call.enqueue(new Callback<ExecutorResponse>() {
            @Override
            public void onResponse(@NotNull Call<ExecutorResponse> call, @NotNull Response<ExecutorResponse> response) {
                ExecutorResponse executorResponse = response.body();
                if (executorResponse != null) {
                    executorId = executorResponse.getExecutorId();
                    executorTopic = executorResponse.getExecutorTopic();
                    trainingTaskParam = executorResponse.getTrainingTaskArgs();
                    if (mFedMlTaskListener != null) {
                        mFedMlTaskListener.onReceive(trainingTaskParam);
                    }
                }
                Log.d(TAG, "onResponse: " + executorResponse);
            }

            @Override
            public void onFailure(@NotNull Call<ExecutorResponse> call, @NotNull Throwable t) {
                Log.w(TAG, "onFailure: ", t);
                if (mFedMlTaskListener != null) {
                    mFedMlTaskListener.onReceive(null);
                }
            }
        });
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
            Log.d(TAG, "Connecting to broker: " + broker);
            client.connect(connOpts);
            client.subscribe("device");
            Log.d(TAG, "Connected");
        } catch (MqttException e) {
            Log.w(TAG, "mqtt create: ", e);
        }

    }

    public void destroy() {
        try {
            client.disconnect();
            Log.d(TAG, "Disconnected");
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
            Log.d(TAG, "连接断开，可以做重连");
            try {
                client.connect(connOpts);
                client.subscribe("device");
            } catch (MqttException e) {
                e.printStackTrace();
            }
            Log.d(TAG, "Connected");
        }

        public void messageArrived(String topic, MqttMessage message) throws Exception {
            // subscribe后得到的消息会执行到这里面
            Log.d(TAG, "接收消息主题:" + topic);
            Log.d(TAG, "接收消息Qos:" + message.getQos());
            Log.d(TAG, "接收消息内容:" + new String(message.getPayload()));
        }

        public void deliveryComplete(IMqttDeliveryToken token) {
            Log.d(TAG, "deliveryComplete---------" + token.isComplete());
        }
    }
}
