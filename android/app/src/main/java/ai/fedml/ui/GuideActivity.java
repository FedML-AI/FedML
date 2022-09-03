package ai.fedml.ui;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;

import java.util.Arrays;

import androidx.annotation.Nullable;

import java.io.File;

import ai.fedml.R;
import ai.fedml.base.AppManager;
import ai.fedml.base.BaseActivity;
import ai.fedml.edge.FedEdgeManager;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.StorageUtils;

/**
 * Guideline pages
 */
public class GuideActivity extends BaseActivity {
    private static final String TAG = "GuideActivity";
    public static final String TRAIN_MODEL_FILE_PATH = StorageUtils.getSdCardPath() + "/ai.fedml/lenet_mnist.mnn";
    public static final String TRAIN_DATA_FILE_PATH = StorageUtils.getSdCardPath() + "/ai.fedml/mnist";
    private final Handler mHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_guide);
        initView();
        loadData();
    }

    private void initView() {
        Log.d(TAG, "guide ModelPath:" + StorageUtils.getModelPath() + "Dataset:" + StorageUtils.getDatasetPath());
        View guideView = findViewById(R.id.iv_guide);
        guideView.setOnClickListener(view -> {
            Log.d(TAG, "OnClick guide ModelPath:" + StorageUtils.getModelPath() +
                    ",Dataset:" + StorageUtils.getDatasetPath());
            Log.d(TAG, "TRAIN_MODEL_FILE_PATH is " + new File(TRAIN_MODEL_FILE_PATH).exists());
            Log.d(TAG, "TRAIN_DATA_FILE_PATH is " + new File(TRAIN_DATA_FILE_PATH).isDirectory());
        });
    }

    private void loadData() {
        mHandler.postDelayed(() -> {
            final String bindingId = FedEdgeManager.getFedEdgeApi().getBoundEdgeId();
            LogHelper.d("BindingId:%s", bindingId);
            if (TextUtils.isEmpty(bindingId)) {
                Intent intent = new Intent();
                intent.setClass(GuideActivity.this, ScanCodeActivity.class);
                startActivity(intent);
            } else {
                Intent intent = new Intent();
                intent.setClass(GuideActivity.this, HomeActivity.class);
                startActivity(intent);
            }
            AppManager.getAppManager().finishActivity();
        }, 200);
    }

    /**
     * Get permission
     */
    private void getPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            int REQUEST_CODE_CONTACT = 101;
            String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE};
            //Verify permission
            for (String str : permissions) {
                if (this.checkSelfPermission(str) != PackageManager.PERMISSION_GRANTED) {
                    //Request permission
                    this.requestPermissions(permissions, REQUEST_CODE_CONTACT);
                }
            }
        }
    }
}
