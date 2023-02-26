package ai.fedml.edge.utils.preference;

import android.content.Context;
import android.content.SharedPreferences;

import ai.fedml.edge.service.ContextHolder;

public class SharePreferencesData {

    private static final String SP_USERINFO = "fedml_Info";

    private static final String KEY_ACCOUNT_ID = "key_account_id";

    private static final String KEY_BINDING_ID = "key_binding_id";

    private static final String KEY_SET_PRIVATE_PATH = "private_path";

    private static final String KEY_HYPER_PARAMETERS = "hyper_parameters";

    /**
     * use SharePreferences save info
     *
     * @param context context
     * @param table   table
     * @param key     key
     * @param value   value
     */
    public static void saveInfo(Context context, String table, String key, String value) {
        SharedPreferences sp = PreferenceUtil.getSharedPreference(context, table);
        SharedPreferences.Editor editor = sp.edit();
        editor.putString(key, value);
        editor.apply();
    }

    /**
     * get info from SharePreferences
     *
     * @param context context
     * @param table   table
     * @param key     KEY
     * @return value
     */
    public static String getInfo(Context context, String table, String key) {
        SharedPreferences sp = PreferenceUtil.getSharedPreference(context, table);
        return sp.getString(key, "");
    }

    /**
     * clear info in SharePreferences
     *
     * @param context context
     * @param table   table
     * @param key     KEY
     */
    public static void clearInfo(Context context, String table, String key) {
        SharedPreferences sp = PreferenceUtil.getSharedPreference(context, table);
        SharedPreferences.Editor editor = sp.edit();
        editor.remove(key);
        editor.apply();
    }

    /**
     * save binding edgeId
     *
     * @param id edgeId
     */
    public static void saveBindingId(final String id) {
        saveInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_BINDING_ID, id);
    }

    /**
     * get binding edgeId
     *
     * @return edgeId
     */
    public static String getBindingId() {
        return getInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_BINDING_ID);
    }

    /**
     * save account id
     *
     * @param id account id
     */
    public static void saveAccountId(final String id) {
        saveInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_ACCOUNT_ID, id);
    }

    /**
     * get account id
     *
     * @return account id
     */
    public static String getAccountId() {
        return getInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_ACCOUNT_ID);
    }

    /**
     * delete binding edgeId
     */
    public static void deleteBindingId() {
        clearInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_BINDING_ID);
    }

    /**
     * save private data path
     *
     * @param path data path
     */
    public static void savePrivatePath(String path) {
        saveInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_SET_PRIVATE_PATH, path);
    }

    /**
     * get private data path
     *
     * @return data path
     */
    public static String getPrivatePath() {
        return getInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_SET_PRIVATE_PATH);
    }

    public static void saveHyperParameters(String hyperParameters) {
        saveInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_HYPER_PARAMETERS, hyperParameters);
    }

    public static String getHyperParameters() {
        return getInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_HYPER_PARAMETERS);
    }

    public static void clearHyperParameters() {
        clearInfo(ContextHolder.getAppContext(), SP_USERINFO, KEY_HYPER_PARAMETERS);
    }
}
