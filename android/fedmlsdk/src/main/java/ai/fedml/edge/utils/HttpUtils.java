package ai.fedml.edge.utils;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class HttpUtils {
    private static final int TIMEOUT_IN_MILLIONS = 15000;
    private static ExecutorService executorService;

    public interface CallBack {
        void onRequestComplete(String result);

        void onRequestFailed();
    }

    public static void init() {
        // 初始化线程池
        int corePoolSize = 10;
        int maximumPoolSize = 20;
        long keepAliveTime = 60;
        TimeUnit unit = TimeUnit.SECONDS;
        BlockingQueue<Runnable> workQueue = new LinkedBlockingQueue<>();
        executorService = new ThreadPoolExecutor(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue);
    }

    public static void doGet(final String urlString, final Map<String, String> params, final CallBack callBack) {
        if (executorService == null) {
            init();
        }

        executorService.execute(new Runnable() {
            @Override
            public void run() {
                HttpURLConnection connection = null;
                try {
                    StringBuilder queryBuilder = new StringBuilder();
                    for (Map.Entry<String, String> entry : params.entrySet()) {
                        String key = entry.getKey();
                        String value = entry.getValue();
                        if (queryBuilder.length() > 0) {
                            queryBuilder.append("&");
                        }
                        queryBuilder.append(URLEncoder.encode(key, "UTF-8"));
                        queryBuilder.append("=");
                        queryBuilder.append(URLEncoder.encode(value, "UTF-8"));
                    }
                    String urlWithQuery = urlString + "?" + queryBuilder;
                    LogHelper.d("urlWithQuery: " + urlWithQuery);
                    URL url = new URL(urlWithQuery);
                    connection = (HttpURLConnection) url.openConnection();
                    connection.setRequestMethod("GET");
                    connection.setConnectTimeout(TIMEOUT_IN_MILLIONS);
                    connection.setReadTimeout(TIMEOUT_IN_MILLIONS);
                    connection.connect();
                    if (connection.getResponseCode() == HttpURLConnection.HTTP_OK) {
                        InputStream is = connection.getInputStream();
                        String result = streamToString(is);
                        if (callBack != null) {
                            callBack.onRequestComplete(result);
                        }
                    } else {
                        LogHelper.w("HttpUtils doGet getResponseCode() is %d", connection.getResponseCode());
                        if (callBack != null) {
                            callBack.onRequestFailed();
                        }
                    }
                } catch (Exception e) {
                    LogHelper.w(e, "HttpUtils doGet failed");
                    if (callBack != null) {
                        callBack.onRequestFailed();
                    }
                } finally {
                    if (connection != null) {
                        connection.disconnect();
                    }
                }
            }
        });
    }

    public static void doPost(final String urlString, final Map<String, String> params, final CallBack callBack) {
        if (executorService == null) {
            init(); // 防止在使用线程池之前尚未初始化
        }

        executorService.execute(new Runnable() {
            @Override
            public void run() {
                HttpURLConnection connection = null;
                OutputStream outputStream = null;
                try {
                    URL url = new URL(urlString);
                    connection = (HttpURLConnection) url.openConnection();
                    connection.setRequestMethod("POST");
                    connection.setConnectTimeout(TIMEOUT_IN_MILLIONS);
                    connection.setReadTimeout(TIMEOUT_IN_MILLIONS);
                    connection.setDoInput(true);
                    connection.setDoOutput(true);
                    connection.setUseCaches(false);
                    StringBuffer sb = new StringBuffer();
                    if (params != null && params.size() > 0) {
                        for (Map.Entry<String, String> entry : params.entrySet()) {
                            sb.append(entry.getKey()).append("=").append(URLEncoder.encode(entry.getValue(), "UTF-8")).append("&");
                        }
                        sb.deleteCharAt(sb.length() - 1);
                    }
                    byte[] data = sb.toString().getBytes();
                    connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
                    connection.setRequestProperty("Content-Length", String.valueOf(data.length));
                    outputStream = connection.getOutputStream();
                    outputStream.write(data);
                    outputStream.flush();

                    int responseCode = connection.getResponseCode();
                    if (responseCode == HttpURLConnection.HTTP_OK) {
                        InputStream is = connection.getInputStream();
                        String result = streamToString(is);
                        if (callBack != null) {
                            callBack.onRequestComplete(result);
                        }
                    } else {
                        LogHelper.w("HttpUtils doPost getResponseCode() is %d", responseCode);
                        callBack.onRequestFailed();
                    }
                } catch (Exception e) {
                    LogHelper.e(e, "HttpUtils doPost failed");
                    callBack.onRequestFailed();
                } finally {
                    if (outputStream != null) {
                        try {
                            outputStream.close();
                        } catch (IOException e) {
                            LogHelper.w(e, "HttpUtils doPost close resource failed");
                        }
                    }
                    if (connection != null) {
                        connection.disconnect();
                    }
                }
            }
        });
    }

    public static void doPost(final String urlString, String requestJson, final CallBack callBack) {
        LogHelper.d("doPost urlString:%s", urlString);

        if (executorService == null) {
            init();
        }

        executorService.execute(new Runnable() {
            @Override
            public void run() {
                HttpURLConnection connection = null;
                OutputStream outputStream = null;
                BufferedWriter writer = null;
                try {
                    URL url = new URL(urlString);
                    connection = (HttpURLConnection) url.openConnection();
                    connection.setConnectTimeout(TIMEOUT_IN_MILLIONS);
                    connection.setReadTimeout(TIMEOUT_IN_MILLIONS);
                    connection.setRequestMethod("POST");
                    connection.setRequestProperty("Connection", "Keep-Alive");
                    connection.setRequestProperty("Charset", "UTF-8");
                    connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
                    connection.setRequestProperty("accept", "application/json");
                    connection.setDoInput(true);
                    connection.setDoOutput(true);
                    connection.setUseCaches(false);
                    connection.connect();

                    writer = new BufferedWriter(new OutputStreamWriter(connection.getOutputStream(), StandardCharsets.UTF_8));
                    writer.write(requestJson);
                    writer.flush();

                    int responseCode = connection.getResponseCode();
                    if (responseCode == HttpURLConnection.HTTP_OK) {
                        InputStream inputStream = connection.getInputStream();
                        String result = streamToString(inputStream);
                        if (null != result && null != callBack) {
                            callBack.onRequestComplete(result);
                        } else {
                            LogHelper.w("HttpUtils doPost failed result is %s", result);
                            callBack.onRequestFailed();
                        }
                    } else {
                        LogHelper.w("HttpUtils doPost getResponseCode() is %d", responseCode);
                        callBack.onRequestFailed();
                    }
                } catch (Exception e) {
                    LogHelper.w(e, "HttpUtils doPost failed");
                    callBack.onRequestFailed();
                } finally {
                    if (writer != null) {
                        try {
                            writer.close();
                        } catch (IOException e) {
                            LogHelper.w(e, "HttpUtils doPost close resource failed");
                        }
                    }

                    if (outputStream != null) {
                        try {
                            outputStream.close();
                        } catch (IOException e) {
                            LogHelper.w(e, "HttpUtils doPost close resource failed");
                        }
                    }
                    if (connection != null) {
                        connection.disconnect();
                    }
                }
            }
        });
    }

    private static String streamToString(InputStream inputStream) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        byte[] bytes = new byte[1024];
        int len;
        try {
            while (-1 != (len = inputStream.read(bytes))) {
                outputStream.write(bytes, 0, len);
            }
            return outputStream.toString();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            try {
                outputStream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }
}
