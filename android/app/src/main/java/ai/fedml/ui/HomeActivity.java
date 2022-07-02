package ai.fedml.ui;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import ai.fedml.R;
import ai.fedml.base.BaseActivity;
import ai.fedml.edge.FedEdgeManager;
import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.service.component.RemoteStorage;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.utils.ToastUtils;
import ai.fedml.widget.CompletedProgressView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

/**
 * HomeActivity
 *
 * @author xkai
 * @date 2021/12/30 14:45
 */
public class HomeActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "HomeActivity";
    private Button btn_set_path;
    private static final int REQUEST_CODE = 1024;
    private TextView mDeviceTextView;
    private TextView mAccLossTextView;
    private CompletedProgressView mProgressView;
    private TextView mHyperTextView;
    private TextView mNameTextView;
    private TextView mEmailTextView;
    private TextView mGroupTextView;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);
        initView();
        loadDate();

    }

    @Override
    protected void onResume() {
        super.onResume();
        String path = FedEdgeManager.getFedEdgeApi().getPrivatePath();
        if (!TextUtils.isEmpty(path)) {
            btn_set_path.setText(path);
        }
    }

    private void initView() {
        btn_set_path = findViewById(R.id.btn_set_path);
        Button btn_unbind = findViewById(R.id.btn_unbind);

        btn_set_path.setOnClickListener(this);
        btn_unbind.setOnClickListener(this);

        mDeviceTextView = findViewById(R.id.tv_device_name);
        mAccLossTextView = findViewById(R.id.tv_acc_loss);
        mProgressView = findViewById(R.id.progress_view);
        mHyperTextView = findViewById(R.id.tv_hyper_parameter);
        mNameTextView = findViewById(R.id.tv_name);
        mEmailTextView = findViewById(R.id.tv_email);
        mGroupTextView = findViewById(R.id.tv_group);
    }

    private void loadDate() {
        requestPermission();
        getUserInfo();
//        VersionUpdate();
        mDeviceTextView.setText(getString(R.string.device_id, FedEdgeManager.getFedEdgeApi().getBoundEdgeId()));
        mProgressView.setProgress(0);
        FedEdgeManager.getFedEdgeApi().setEpochLossListener(new OnTrainProgressListener() {
            private int mRound = 0;
            private int mEpoch = 0;
            private float mLoss = 0f;
            private float mAccuracy = 0f;

            @Override
            public void onEpochLoss(int round, int epoch, float loss) {
                mRound = round;
                mEpoch = epoch;
                mLoss = loss;
                runOnUiThread(() ->
                        mAccLossTextView.setText(getString(R.string.acc_loss_txt, mRound, mEpoch, mAccuracy, mLoss)));
            }

            @Override
            public void onEpochAccuracy(int round, int epoch, float accuracy) {
                mRound = round;
                mEpoch = epoch;
                mAccuracy = accuracy;
                runOnUiThread(() ->
                        mAccLossTextView.setText(getString(R.string.acc_loss_txt, mRound, mEpoch, mAccuracy, mLoss)));
            }

            @Override
            public void onProgressChanged(int round, int progress) {
                runOnUiThread(() ->
                        mProgressView.setProgress(progress));
            }
        });
        FedEdgeManager.getFedEdgeApi().setTrainingStatusListener((status) ->
                runOnUiThread(() -> mHyperTextView.setText(FedEdgeManager.getFedEdgeApi().getHyperParameters())));
    }


    @SuppressLint("NonConstantResourceId")
    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_set_path:
                Intent intent = new Intent();
                intent.setClass(HomeActivity.this, SetFilePathActivity.class);
                startActivity(intent);
                break;
            case R.id.btn_unbind:
                unbound();
                break;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // 先判断有没有权限
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + this.getPackageName()));
                startActivityForResult(intent, REQUEST_CODE);
            }
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            // 先判断有没有权限
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ||
                    ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CODE);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            } else {
                ToastUtils.show("EXTERNAL STORAGE Permissions failed");
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (Environment.isExternalStorageManager()) {
            } else {
                ToastUtils.show("存储权限获取失败");
            }
        }
    }

    private void getUserInfo() {
        RequestManager.getUserInfo(userInfo -> {
            if (userInfo != null) {
                runOnUiThread(() -> {
                    mNameTextView.setText(String.format("%s %s", userInfo.getLastname(), userInfo.getFirstName()));
                    mEmailTextView.setText(userInfo.getEmail());
                    mGroupTextView.setText(userInfo.getCompany());
                });
            }
        });
    }


    private void unbound() {
        String bindingId = FedEdgeManager.getFedEdgeApi().getBoundEdgeId();
        LogHelper.d("unbound bindingId:%s", bindingId);
        RequestManager.unboundAccount(bindingId, isSuccess -> runOnUiThread(() -> {
            if (isSuccess) {
                // 跳转至扫描页
                Intent intent = new Intent();
                intent.setClass(HomeActivity.this, ScanCodeActivity.class);
                startActivity(intent);
                finish();
            }
        }));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}
