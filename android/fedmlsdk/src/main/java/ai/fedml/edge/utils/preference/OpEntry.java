package ai.fedml.edge.utils.preference;

import android.os.Bundle;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import androidx.annotation.NonNull;

class OpEntry {

    static final int OP_TYPE_GET = 1;

    static final int OP_TYPE_PUT = 2;

    static final int OP_TYPE_CLEAR = 3;

    static final int OP_TYPE_REMOVE = 4;

    static final int OP_TYPE_COMMIT = 5;

    static final int OP_TYPE_APPLY = 6;


    static final int VALUE_TYPE_STRING = 1;

    static final int VALUE_TYPE_INT = 2;

    static final int VALUE_TYPE_LONG = 3;

    static final int VALUE_TYPE_FLOAT = 4;

    static final int VALUE_TYPE_BOOLEAN = 5;

    static final int VALUE_TYPE_STRING_SET = 6;


    static final String KEY_KEY = "key_key";

    static final String KEY_VALUE = "key_value";


    static final String KEY_VALUE_TYPE = "key_value_type";

    static final String KEY_OP_TYPE = "key_op_type";

    @NonNull
    private Bundle bundle;

    private OpEntry() {
        this.bundle = new Bundle();
    }

    public OpEntry(@NonNull Bundle bundle) {
        this.bundle = bundle;
    }

    public String getKey() {
        return bundle.getString(KEY_KEY, null);
    }

    public OpEntry setKey(String key) {
        bundle.putString(KEY_KEY, key);
        return this;
    }

    public int getValueType() {
        return bundle.getInt(KEY_VALUE_TYPE, 0);
    }

    public OpEntry setValueType(int valueType) {
        bundle.putInt(KEY_VALUE_TYPE, valueType);
        return this;
    }

    public int getOpType() {
        return bundle.getInt(KEY_OP_TYPE, 0);
    }

    public OpEntry setOpType(int opType) {
        bundle.putInt(KEY_OP_TYPE, opType);
        return this;
    }

    public String getStringValue(String defValue) {
        return bundle.getString(KEY_VALUE, defValue);
    }

    public OpEntry setStringValue(String value) {
        bundle.putInt(KEY_VALUE_TYPE, VALUE_TYPE_STRING);
        bundle.putString(KEY_VALUE, value);
        return this;
    }

    public int getIntValue(int defValue) {
        return bundle.getInt(KEY_VALUE, defValue);
    }

    public OpEntry setIntValue(int value) {
        bundle.putInt(KEY_VALUE_TYPE, VALUE_TYPE_INT);
        bundle.putInt(KEY_VALUE, value);
        return this;
    }

    public long getLongValue(long defValue) {
        return bundle.getLong(KEY_VALUE, defValue);
    }

    public OpEntry setLongValue(long value) {
        bundle.putInt(KEY_VALUE_TYPE, VALUE_TYPE_LONG);
        bundle.putLong(KEY_VALUE, value);
        return this;
    }

    public float getFloatValue(float defValue) {
        return bundle.getFloat(KEY_VALUE);
    }

    public OpEntry setFloatValue(float value) {
        bundle.putInt(KEY_VALUE_TYPE, VALUE_TYPE_FLOAT);
        bundle.putFloat(KEY_VALUE, value);
        return this;
    }


    public boolean getBooleanValue(boolean defValue) {
        return bundle.getBoolean(KEY_VALUE, defValue);
    }

    public OpEntry setBooleanValue(boolean value) {
        bundle.putInt(KEY_VALUE_TYPE, VALUE_TYPE_BOOLEAN);
        bundle.putBoolean(KEY_VALUE, value);
        return this;
    }

    public Set<String> getStringSet() {
        ArrayList<String> list = bundle.getStringArrayList(KEY_VALUE);
        return list == null ? null : new HashSet<>(list);
    }


    @NonNull
    public Bundle getBundle() {
        return bundle;
    }

    public OpEntry setStringSettingsValue(Set<String> value) {
        bundle.putInt(KEY_VALUE_TYPE, VALUE_TYPE_STRING_SET);
        bundle.putStringArrayList(KEY_VALUE, value == null ? null : new ArrayList<>(value));
        return this;
    }


    static OpEntry obtainGetOperation(String key) {
        return new OpEntry()
                .setKey(key)
                .setOpType(OP_TYPE_GET);
    }

    static OpEntry obtainPutOperation(String key) {
        return new OpEntry()
                .setKey(key)
                .setOpType(OP_TYPE_PUT);
    }

    static OpEntry obtainRemoveOperation(String key) {
        return new OpEntry()
                .setKey(key)
                .setOpType(OP_TYPE_REMOVE);
    }

    static OpEntry obtainClear() {
        return new OpEntry()
                .setOpType(OP_TYPE_CLEAR);
    }
}
