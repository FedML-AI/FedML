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

import ai.fedml.edge.nativemnn.NativeFedMLTrainer;
import ai.fedml.edge.nativemnn.TrainingCallback;
import androidx.annotation.Nullable;

import java.io.File;

import ai.fedml.R;
import ai.fedml.base.AppManager;
import ai.fedml.base.BaseActivity;
import ai.fedml.edge.FedEdgeManager;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.StorageUtils;

/**
 * 引导页
 *
 * @author xkai
 * @date 2021/12/30
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
        getPermission();
        initView();
//        loadData();
    }

    private void initView() {
        Log.d(TAG, "guide ModelPath:" + StorageUtils.getModelPath() + "Dataset:" + StorageUtils.getDatasetPath());
        View guideView = findViewById(R.id.iv_guide);
        guideView.setOnClickListener(view -> {
            Log.d(TAG, "OnClick guide ModelPath:" + StorageUtils.getModelPath() +
                    ",Dataset:" + StorageUtils.getDatasetPath());
            Log.d(TAG, "TRAIN_MODEL_FILE_PATH is " + new File(TRAIN_MODEL_FILE_PATH).exists());
            Log.d(TAG, "TRAIN_DATA_FILE_PATH is " + new File(TRAIN_DATA_FILE_PATH).isDirectory());
            testTrain();
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
     * 获取权限
     */
    private void getPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            int REQUEST_CODE_CONTACT = 101;
            String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE};
            //验证是否许可权限
            for (String str : permissions) {
                if (this.checkSelfPermission(str) != PackageManager.PERMISSION_GRANTED) {
                    //申请权限
                    this.requestPermissions(permissions, REQUEST_CODE_CONTACT);
                }
            }
        }
    }

    private void testTrain() {
        //test parameter for the encoding part
        final String dataSetType = "mnist";
        final String modelPath = TRAIN_MODEL_FILE_PATH;
        final String dataPath = TRAIN_DATA_FILE_PATH;

        int client_num = 10;
        int q_bits = 15;
        int p = (int) (Math.pow(2, 15) - 19);
        final int batchSizeNum = 128;
        double learningRate = 0.1;
        final int epochNum = 1;
        int trainSize = 50000;
        int testSize = 10000;

        final TrainingCallback callback = new TrainingCallback() {
            @Override
            public void onProgress(float progress) {
                LogHelper.d("onProgress(%f)", progress);
            }

            @Override
            public void onAccuracy(int epoch, float accuracy) {
                LogHelper.d("onAccuracy(%d, %f)", epoch, accuracy);
            }

            @Override
            public void onLoss(int epoch, float loss) {
                LogHelper.d("onLoss(%d, %f)", epoch, loss);
            }
        };
        NativeFedMLTrainer trainer = new NativeFedMLTrainer();
        //init all required parameters
        trainer.init(modelPath, dataPath, dataSetType, trainSize, testSize, batchSizeNum, learningRate,
                epochNum, q_bits, p, client_num, callback);
        // 1. generate mask and encode local mask for others
        float[][] maskMatrix = trainer.getLocalEncodedMask();
        LogHelper.d("getLocalEncodedMask maskMatrix=%s", Arrays.toString(maskMatrix[0]));
        LogHelper.d("getLocalEncodedMask maskMatrix=%d", maskMatrix[0].length);
        // 2. share and receive local mask from others via server (including share to self)
        int client_index = 1;
        float[] local_encode_mask = maskMatrix[0];
        trainer.saveMaskFromPairedClients(client_index, local_encode_mask);

        int client_index_another = 3;
        float[] local_encode_mask_another = maskMatrix[0];
        trainer.saveMaskFromPairedClients(client_index_another, local_encode_mask_another);
        // 3. report receive online users to server
        int [] online_user = trainer.getClientIdsThatHaveSentMask();
        LogHelper.d("online_user length=%d", online_user.length);
        LogHelper.d("online_user=%d", online_user[0]);
        LogHelper.d("online_user=%d", online_user[1]);

        // 4. do training
        String result = trainer.train();
        LogHelper.d("online_user=%s", result);
//        // 5. save masked model
//        trainer.generateMaskedModel();
        // 6. aggregate received mask
        int[] surviving_list;
        surviving_list = new int[2];
        surviving_list[0] = 1;
        surviving_list[1] = 3;
        float[] agg_mask = trainer.getAggregatedEncodedMask(surviving_list);
        LogHelper.d("agg mask=%s", Arrays.toString(agg_mask));
    }

}
