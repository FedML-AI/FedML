package ai.fedml.edge.utils.preference;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Process;
import android.util.ArrayMap;

import java.util.ArrayList;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

class SharedPreferenceProxy implements SharedPreferences {
    private static Map<String, SharedPreferenceProxy> sharedPreferenceProxyMap;

    /**
     * Flag whether caller process is the same with provider
     * 0: unknown
     * 1: the same
     * -1: not the same
     */
    private static final AtomicInteger processFlag = new AtomicInteger(0);

    private final Context ctx;
    private final String preferName;

    private SharedPreferenceProxy(Context ctx, String name) {
        this.ctx = ctx.getApplicationContext();
        this.preferName = name;
    }

    @Override
    public Map<String, ?> getAll() {
        throw new UnsupportedOperationException("Not support method getAll");
    }

    @Nullable
    @Override
    public String getString(String key, @Nullable String defValue) {
        OpEntry result = getResult(OpEntry.obtainGetOperation(key).setStringValue(defValue));
        return result == null ? defValue : result.getStringValue(defValue);
    }

    @Nullable
    @Override
    public Set<String> getStringSet(String key, @Nullable Set<String> defValues) {
        OpEntry result = getResult(OpEntry.obtainGetOperation(key).setStringSettingsValue(defValues));
        if (result == null) {
            return defValues;
        }
        Set<String> set = result.getStringSet();
        if (set == null) {
            return defValues;
        }
        return set;
    }

    @Override
    public int getInt(String key, int defValue) {
        OpEntry result = getResult(OpEntry.obtainGetOperation(key).setIntValue(defValue));
        return result == null ? defValue : result.getIntValue(defValue);
    }

    @Override
    public long getLong(String key, long defValue) {
        OpEntry result = getResult(OpEntry.obtainGetOperation(key).setLongValue(defValue));
        return result == null ? defValue : result.getLongValue(defValue);
    }

    @Override
    public float getFloat(String key, float defValue) {
        OpEntry result = getResult(OpEntry.obtainGetOperation(key).setFloatValue(defValue));
        return result == null ? defValue : result.getFloatValue(defValue);
    }

    @Override
    public boolean getBoolean(String key, boolean defValue) {
        OpEntry result = getResult(OpEntry.obtainGetOperation(key).setBooleanValue(defValue));
        return result == null ? defValue : result.getBooleanValue(defValue);
    }

    @Override
    public boolean contains(String key) {
        Bundle input = new Bundle();
        input.putString(OpEntry.KEY_KEY, key);
        try {
            Bundle res = ctx.getContentResolver().call(PreferenceUtil.URI
                    , PreferenceUtil.METHOD_CONTAIN_KEY
                    , preferName
                    , input);
            return res.getBoolean(PreferenceUtil.KEY_VALUES);
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    @Override
    public Editor edit() {
        return new EditorImpl();
    }

    @Override
    public void registerOnSharedPreferenceChangeListener(OnSharedPreferenceChangeListener listener) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void unregisterOnSharedPreferenceChangeListener(OnSharedPreferenceChangeListener listener) {
        throw new UnsupportedOperationException();
    }

    private OpEntry getResult(@NonNull OpEntry input) {
        try {
            Bundle res = ctx.getContentResolver().call(PreferenceUtil.URI
                    , PreferenceUtil.METHOD_QUERY_VALUE
                    , preferName
                    , input.getBundle());
            return new OpEntry(res);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public class EditorImpl implements Editor {
        private final ArrayList<OpEntry> mModified = new ArrayList<>();

        @Override
        public Editor putString(String key, @Nullable String value) {
            OpEntry entry = OpEntry.obtainPutOperation(key).setStringValue(value);
            return addOps(entry);
        }

        @Override
        public Editor putStringSet(String key, @Nullable Set<String> values) {
            OpEntry entry = OpEntry.obtainPutOperation(key).setStringSettingsValue(values);
            return addOps(entry);
        }

        @Override
        public Editor putInt(String key, int value) {
            OpEntry entry = OpEntry.obtainPutOperation(key).setIntValue(value);
            return addOps(entry);
        }

        @Override
        public Editor putLong(String key, long value) {
            OpEntry entry = OpEntry.obtainPutOperation(key).setLongValue(value);
            return addOps(entry);
        }

        @Override
        public Editor putFloat(String key, float value) {
            OpEntry entry = OpEntry.obtainPutOperation(key).setFloatValue(value);
            return addOps(entry);
        }

        @Override
        public Editor putBoolean(String key, boolean value) {
            OpEntry entry = OpEntry.obtainPutOperation(key).setBooleanValue(value);
            return addOps(entry);
        }

        @Override
        public Editor remove(String key) {
            OpEntry entry = OpEntry.obtainRemoveOperation(key);
            return addOps(entry);
        }

        @Override
        public Editor clear() {
            return addOps(OpEntry.obtainClear());
        }

        @Override
        public boolean commit() {
            Bundle input = new Bundle();
            input.putParcelableArrayList(PreferenceUtil.KEY_VALUES, convertBundleList());
            input.putInt(OpEntry.KEY_OP_TYPE, OpEntry.OP_TYPE_COMMIT);
            Bundle res = null;
            try {
                res = ctx.getContentResolver().call(PreferenceUtil.URI, PreferenceUtil.METHOD_EDIT_VALUE, preferName, input);
            } catch (Exception e) {
                e.printStackTrace();
            }
            if (res == null) {
                return false;
            }
            return res.getBoolean(PreferenceUtil.KEY_VALUES, false);
        }

        @Override
        public void apply() {
            Bundle input = new Bundle();
            input.putParcelableArrayList(PreferenceUtil.KEY_VALUES, convertBundleList());
            input.putInt(OpEntry.KEY_OP_TYPE, OpEntry.OP_TYPE_APPLY);
            try {
                ctx.getContentResolver().call(PreferenceUtil.URI, PreferenceUtil.METHOD_EDIT_VALUE, preferName, input);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private Editor addOps(OpEntry op) {
            synchronized (this) {
                mModified.add(op);
                return this;
            }
        }

        private ArrayList<Bundle> convertBundleList() {
            synchronized (this) {
                ArrayList<Bundle> bundleList = new ArrayList<>(mModified.size());
                for (OpEntry entry : mModified) {
                    bundleList.add(entry.getBundle());
                }
                return bundleList;
            }
        }
    }





    public static SharedPreferences getSharedPreferences(@NonNull Context ctx, String preferName) {
        //First check if the same process
        if (processFlag.get() == 0) {
            Bundle bundle = ctx.getContentResolver().call(PreferenceUtil.URI, PreferenceUtil.METHOD_QUERY_PID, "", null);
            int pid = 0;
            if (bundle != null) {
                pid = bundle.getInt(PreferenceUtil.KEY_VALUES);
            }
            //Can not get the pid, something wrong!
            if (pid == 0) {
                return getFromLocalProcess(ctx, preferName);
            }
            processFlag.set(Process.myPid() == pid ? 1 : -1);
            return getSharedPreferences(ctx, preferName);
        } else if (processFlag.get() > 0) {
            return getFromLocalProcess(ctx, preferName);
        } else {
            return getFromRemoteProcess(ctx, preferName);
        }
    }


    private static SharedPreferences getFromRemoteProcess(@NonNull Context ctx, String preferName) {
        synchronized (SharedPreferenceProxy.class) {
            if (sharedPreferenceProxyMap == null) {
                sharedPreferenceProxyMap = new ArrayMap<>();
            }
            SharedPreferenceProxy preferenceProxy = sharedPreferenceProxyMap.get(preferName);
            if (preferenceProxy == null) {
                preferenceProxy = new SharedPreferenceProxy(ctx.getApplicationContext(), preferName);
                sharedPreferenceProxyMap.put(preferName, preferenceProxy);
            }
            return preferenceProxy;
        }
    }

    private static SharedPreferences getFromLocalProcess(@NonNull Context ctx, String preferName) {
        return ctx.getSharedPreferences(preferName, Context.MODE_PRIVATE);
    }
}
