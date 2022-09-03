package ai.fedml.edge.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.CharArrayReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;

public class GsonUtils {
    private static final Gson sGson = new GsonBuilder().create();

    public static String toJson(Object obj) {
        return sGson.toJson(obj);
    }

    public static <T> T fromJson(String json, Class<T> classOfT) throws JsonSyntaxException {
        return sGson.fromJson(json, classOfT);
    }

    public static <T> T fromJson(byte[] bytes, Class<T> classOfT) throws JsonSyntaxException {
        try (InputStream inputStream = new ByteArrayInputStream(bytes);
             Reader reader = new InputStreamReader(inputStream)) {
            return sGson.fromJson(reader, classOfT);
        } catch (IOException e) {
            LogHelper.e(e, "fromJson Exception.");
        }
        return null;
    }

    public static <T> T fromJson(InputStream inputStream, Class<T> classOfT) throws JsonSyntaxException {
        try (Reader reader = new InputStreamReader(inputStream)) {
            return sGson.fromJson(reader, classOfT);
        } catch (IOException e) {
            LogHelper.e(e, "fromJson Exception.");
        }
        return null;
    }
}
