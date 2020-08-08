package ai.fedml.iot.http.simplehttp;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public class SimpleHttp {
    public final static String BASE_URL_IOT_APP_SERVICE = "http://app.iotsupercloud.com";
    //public final static String BASE_URL_IOT_APP_SERVICE = "http://111.230.226.28";
    public interface HttpGetCallback{
        void onResponse(String strResponse);
        void onFailure(Throwable throwable);
    }
    public static void get(String strUrl, HttpGetCallback callback){
        BufferedReader in = null;
        StringBuilder result = new StringBuilder();
        try {
            URL url = new URL(strUrl);
            HttpURLConnection httpURLConnection = (HttpURLConnection) url.openConnection();
            System.out.println("debug1");

            httpURLConnection.setRequestMethod("GET");
            //Get请求不需要DoOutPut
            httpURLConnection.setDoOutput(false);
            httpURLConnection.setDoInput(true);
            //设置连接超时时间和读取超时时间
            httpURLConnection.setConnectTimeout(10000);
            httpURLConnection.setReadTimeout(10000);
            httpURLConnection.setInstanceFollowRedirects(false);

            //连接服务器
            httpURLConnection.connect();
            System.out.println("responseCode = " + httpURLConnection.getResponseCode());

            String location = httpURLConnection.getHeaderField("Location");
            System.out.println("redirected location = " + location);

            // 取得输入流，并使用Reader读取
            in = new BufferedReader(new InputStreamReader(httpURLConnection.getInputStream(), "UTF-8"));
            String line;
            while ((line = in.readLine()) != null) {
                result.append(line);
            }
            if(callback != null){
                callback.onResponse(result.toString());
            }
        } catch (MalformedURLException e) {
            callback.onFailure(e);
        } catch (IOException e) {
            callback.onFailure(e);
        }
    }

    public static void main(String args[]){
        String url = BASE_URL_IOT_APP_SERVICE + "/iotdevice/" + "chaoyanghe";
        SimpleHttp.get(url, new HttpGetCallback() {
            @Override
            public void onResponse(String strResponse) {
                System.out.println(strResponse);
            }

            @Override
            public void onFailure(Throwable throwable) {
                throwable.printStackTrace();
                System.out.println(throwable.getMessage());
            }
        });
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
