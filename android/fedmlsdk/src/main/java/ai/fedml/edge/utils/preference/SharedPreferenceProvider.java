package ai.fedml.edge.utils.preference;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Context;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Process;
import android.util.ArrayMap;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;


public class SharedPreferenceProvider extends ContentProvider{

    private final Map<String, MethodProcess> processorMap = new ArrayMap<>();
    @Override
    public boolean onCreate() {
        processorMap.put(PreferenceUtil.METHOD_QUERY_VALUE, methodQueryValues);
        processorMap.put(PreferenceUtil.METHOD_CONTAIN_KEY, methodContainKey);
        processorMap.put(PreferenceUtil.METHOD_EDIT_VALUE, methodEditor);
        processorMap.put(PreferenceUtil.METHOD_QUERY_PID, methodQueryPid);
        return true;
    }

    @Nullable
    @Override
    public Cursor query(@NonNull Uri uri, @Nullable String[] projection, @Nullable String selection, @Nullable String[] selectionArgs, @Nullable String sortOrder) {
        throw new UnsupportedOperationException();
    }

    @Nullable
    @Override
    public String getType(@NonNull Uri uri) {
        throw new UnsupportedOperationException();
    }

    @Nullable
    @Override
    public Uri insert(@NonNull Uri uri, @Nullable ContentValues values) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int delete(@NonNull Uri uri, @Nullable String selection, @Nullable String[] selectionArgs) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int update(@NonNull Uri uri, @Nullable ContentValues values, @Nullable String selection, @Nullable String[] selectionArgs) {
        throw new UnsupportedOperationException();
    }

    @Nullable
    @Override
    public Bundle call(@NonNull String method, @Nullable String arg, @Nullable Bundle extras) {
        MethodProcess processor = processorMap.get(method);
        return processor == null?null:processor.process(arg, extras);
    }


    public interface MethodProcess {
        Bundle process(@Nullable String arg, @Nullable Bundle extras);
    }

    private final MethodProcess methodQueryPid = (arg, extras) -> {
        Bundle bundle = new Bundle();
        bundle.putInt(PreferenceUtil.KEY_VALUES, Process.myPid());
        return bundle;
    };

    private final MethodProcess methodQueryValues = (arg, extras) -> {
        if (extras == null) {
            throw new IllegalArgumentException("methodQueryValues, extras is null!");
        }
        Context ctx = getContext();
        if (ctx == null) {
            throw new IllegalArgumentException("methodQueryValues, ctx is null!");
        }
        String key = extras.getString(OpEntry.KEY_KEY);
        SharedPreferences preferences = ctx.getSharedPreferences(arg, Context.MODE_PRIVATE);
        int valueType = extras.getInt(OpEntry.KEY_VALUE_TYPE);
        switch (valueType) {
            case OpEntry.VALUE_TYPE_BOOLEAN:{
                boolean value = preferences.getBoolean(key, extras.getBoolean(OpEntry.KEY_VALUE));
                extras.putBoolean(OpEntry.KEY_VALUE, value);
                return extras;
            }
            case OpEntry.VALUE_TYPE_FLOAT:{
                float value = preferences.getFloat(key, extras.getFloat(OpEntry.KEY_VALUE));
                extras.putFloat(OpEntry.KEY_VALUE, value);
                return extras;
            }
            case OpEntry.VALUE_TYPE_INT:{
                int value = preferences.getInt(key, extras.getInt(OpEntry.KEY_VALUE));
                extras.putInt(OpEntry.KEY_VALUE, value);
                return extras;
            }
            case OpEntry.VALUE_TYPE_LONG:{
                long value = preferences.getLong(key, extras.getLong(OpEntry.KEY_VALUE));
                extras.putLong(OpEntry.KEY_VALUE, value);
                return extras;
            }
            case OpEntry.VALUE_TYPE_STRING:{
                String value = preferences.getString(key, extras.getString(OpEntry.KEY_VALUE));
                extras.putString(OpEntry.KEY_VALUE, value);
                return extras;
            }
            case OpEntry.VALUE_TYPE_STRING_SET:{
                Set<String> value = preferences.getStringSet(key, null);
                extras.putStringArrayList(OpEntry.KEY_VALUE, value == null?null:new ArrayList<>(value));
                return extras;
            }
            default:{
                throw new IllegalArgumentException("unknown valueType:" + valueType);
            }
        }
    };

    private final MethodProcess methodContainKey = (arg, extras) -> {
        if (extras == null) {
            throw new IllegalArgumentException("methodQueryValues, extras is null!");
        }
        Context ctx = getContext();
        if (ctx == null) {
            throw new IllegalArgumentException("methodQueryValues, ctx is null!");
        }
        String key = extras.getString(OpEntry.KEY_KEY);
        SharedPreferences preferences = ctx.getSharedPreferences(arg, Context.MODE_PRIVATE);
        extras.putBoolean(PreferenceUtil.KEY_VALUES, preferences.contains(key));
        return extras;
    };

    private final MethodProcess methodEditor = new MethodProcess() {
        @Override
        public Bundle process(@Nullable String arg, @Nullable Bundle extras) {
            if (extras == null) {
                throw new IllegalArgumentException("methodQueryValues, extras is null!");
            }
            Context ctx = getContext();
            if (ctx == null) {
                throw new IllegalArgumentException("methodQueryValues, ctx is null!");
            }
            SharedPreferences preferences = ctx.getSharedPreferences(arg, Context.MODE_PRIVATE);
            ArrayList<Bundle> ops = extras.getParcelableArrayList(PreferenceUtil.KEY_VALUES);
            if (ops == null) {
                ops = new ArrayList<>();
            }
            SharedPreferences.Editor editor = preferences.edit();
            for (Bundle opBundler : ops) {
                int opType = opBundler.getInt(OpEntry.KEY_OP_TYPE);
                switch (opType) {
                    case OpEntry.OP_TYPE_PUT: {
                        editor = editValue(editor, opBundler);
                        break;
                    }
                    case OpEntry.OP_TYPE_REMOVE: {
                        editor = editor.remove(opBundler.getString(OpEntry.KEY_KEY));
                        break;
                    }
                    case OpEntry.OP_TYPE_CLEAR: {
                        editor = editor.clear();
                        break;
                    }
                    default: {
                        throw new IllegalArgumentException("unknown op type:" + opType);
                    }
                }
            }

            int applyOrCommit = extras.getInt(OpEntry.KEY_OP_TYPE);
            if (applyOrCommit == OpEntry.OP_TYPE_APPLY) {
                editor.apply();
                return null;
            } else if (applyOrCommit == OpEntry.OP_TYPE_COMMIT) {
                boolean res = editor.commit();
                Bundle bundle = new Bundle();
                bundle.putBoolean(PreferenceUtil.KEY_VALUES, res);
                return bundle;
            } else {
                throw new IllegalArgumentException("unknown applyOrCommit:" + applyOrCommit);
            }
        }


        private SharedPreferences.Editor editValue(SharedPreferences.Editor editor, Bundle opBundle) {
            String key = opBundle.getString(OpEntry.KEY_KEY);
            int valueType = opBundle.getInt(OpEntry.KEY_VALUE_TYPE);
            switch (valueType) {
                case OpEntry.VALUE_TYPE_BOOLEAN: {
                    return editor.putBoolean(key, opBundle.getBoolean(OpEntry.KEY_VALUE));
                }
                case OpEntry.VALUE_TYPE_FLOAT: {
                    return editor.putFloat(key, opBundle.getFloat(OpEntry.KEY_VALUE));
                }
                case OpEntry.VALUE_TYPE_INT: {
                    return editor.putInt(key, opBundle.getInt(OpEntry.KEY_VALUE));
                }
                case OpEntry.VALUE_TYPE_LONG: {
                    return editor.putLong(key, opBundle.getLong(OpEntry.KEY_VALUE));
                }
                case OpEntry.VALUE_TYPE_STRING: {
                    return editor.putString(key, opBundle.getString(OpEntry.KEY_VALUE));
                }
                case OpEntry.VALUE_TYPE_STRING_SET: {
                    ArrayList<String> list = opBundle.getStringArrayList(OpEntry.KEY_VALUE);
                    if (list == null) {
                        return editor.putStringSet(key, null);
                    }
                    return editor.putStringSet(key, new HashSet<>(list));
                }
                default: {
                    throw new IllegalArgumentException("unknown valueType:" + valueType);
                }
            }
        }
    };
}
