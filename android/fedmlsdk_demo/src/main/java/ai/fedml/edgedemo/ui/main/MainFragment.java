package ai.fedml.edgedemo.ui.main;

import ai.fedml.edge.FedEdgeManager;
import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edgedemo.App;
import ai.fedml.edgedemo.GlideApp;
import ai.fedml.edgedemo.widget.CompletedProgressView;
import androidx.lifecycle.ViewModelProvider;

import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import ai.fedml.edgedemo.R;

public class MainFragment extends Fragment {

    private MainViewModel mViewModel;
    private TextView mStatusTextView;
    private TextView mAccLossTextView;
    private CompletedProgressView mProgressView;
    private TextView mHyperTextView;
    private TextView mNameTextView;
    private TextView mEmailTextView;
    private TextView mGroupTextView;
    private ImageView mAvatarImageView;
    private TextView mDeviceAccountInfoTextView;
    private TextView mUnInitButton;
    private TextView mInitButton;

    public static MainFragment newInstance() {
        return new MainFragment();
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        mViewModel = new ViewModelProvider(this).get(MainViewModel.class);
        View view = inflater.inflate(R.layout.main_fragment, container, false);
        initView(view);
        loadData();
        return view;
    }

    private void initView(final @NonNull View view) {
        mStatusTextView = view.findViewById(R.id.tv_status);
        mAccLossTextView = view.findViewById(R.id.tv_acc_loss);
        mProgressView = view.findViewById(R.id.progress_view);
        mHyperTextView = view.findViewById(R.id.tv_hyper_parameter);
        mDeviceAccountInfoTextView = view.findViewById(R.id.tv_account_info);
        mNameTextView = view.findViewById(R.id.tv_name);
        mEmailTextView = view.findViewById(R.id.tv_email);
        mGroupTextView = view.findViewById(R.id.tv_group);
        mAvatarImageView = view.findViewById(R.id.iv_avatar);
//        mUnInitButton = view.findViewById(R.id.btn_uninit);
//        mInitButton = view.findViewById(R.id.btn_init);
    }

    private void loadData() {
//        mInitButton.setOnClickListener((view)->{
//            FedEdgeManager.getFedEdgeApi().init(getContext());
//        });
//        mUnInitButton.setOnClickListener((view) -> {
//            FedEdgeManager.getFedEdgeApi().unInit();
//        });
        getUserInfo();
        mDeviceAccountInfoTextView.setText(getString(R.string.account_information, FedEdgeManager.getFedEdgeApi().getBoundEdgeId()));
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
                App.runOnUiThread(() ->
                        mAccLossTextView.setText(getString(R.string.acc_loss_txt, mRound, mEpoch, mAccuracy, mLoss)));
            }

            @Override
            public void onEpochAccuracy(int round, int epoch, float accuracy) {
                mRound = round;
                mEpoch = epoch;
                mAccuracy = accuracy;
                App.runOnUiThread(() ->
                        mAccLossTextView.setText(getString(R.string.acc_loss_txt, mRound, mEpoch, mAccuracy, mLoss)));
            }

            @Override
            public void onProgressChanged(int round, float progress) {
                App.runOnUiThread(() ->
                        mProgressView.setProgress(Math.round(progress)));
            }
        });
        FedEdgeManager.getFedEdgeApi().setTrainingStatusListener((status) ->
                App.runOnUiThread(() -> {
                    Log.d("setTrainingStatusListener", "FedMLDebug status = " + status);
                    if (status == MessageDefine.KEY_CLIENT_STATUS_INITIALIZING ||
                            status == MessageDefine.KEY_CLIENT_STATUS_KILLED ||
                            status == MessageDefine.KEY_CLIENT_STATUS_IDLE ||
                            status == MessageDefine.KEY_CLIENT_STATUS_FAILED) {
                        LogHelper.d("FedEdgeManager", "FedMLDebug. status = " + status);
                        mHyperTextView.setText(FedEdgeManager.getFedEdgeApi().getHyperParameters());
                        mProgressView.setProgress(0);
                        mAccLossTextView.setText(getString(R.string.acc_loss_txt, 0, 0, 0.0, 0.0));
                    }
                    mStatusTextView.setText(MessageDefine.CLIENT_STATUS_MAP.get(status));

                }));
    }

    private void getUserInfo() {
        FedEdgeManager.getFedEdgeApi().getUserInfo(userInfo -> {
            if (userInfo != null) {
                App.runOnUiThread(() -> {
                    mNameTextView.setText(String.format("%s %s", userInfo.getLastname(), userInfo.getFirstName()));
                    mEmailTextView.setText(userInfo.getEmail());
                    mGroupTextView.setText(userInfo.getCompany());
                    GlideApp.with(MainFragment.this)
                            .load(userInfo.getAvatar())
                            .circleCrop()
                            .error(R.mipmap.ic_shijiali)
                            .placeholder(R.mipmap.ic_shijiali)
                            .into(mAvatarImageView);
                });
            }
        });
    }
}