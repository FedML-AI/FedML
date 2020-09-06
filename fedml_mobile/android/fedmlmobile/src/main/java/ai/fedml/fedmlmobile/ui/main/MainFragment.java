package ai.fedml.fedmlmobile.ui.main;

import androidx.lifecycle.ViewModelProvider;

import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import ai.fedml.fedmlmobile.R;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import butterknife.Unbinder;

public class MainFragment extends Fragment {

    public static MainFragment newInstance() {
        return new MainFragment();
    }

    private MainViewModel mViewModel;
    private Unbinder unbinder;
    @BindView(R.id.message)
    TextView messageTextView;
    @BindView(R.id.btn_ok)
    Button okButton;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.main_fragment, container, false);
        unbinder = ButterKnife.bind(this, view);
        return view;
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        mViewModel = new ViewModelProvider(this).get(MainViewModel.class);

        // TODO: Use the ViewModel
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        unbinder.unbind();
    }

    @OnClick(R.id.btn_ok)
    void initial() {
        mViewModel.initial(getContext());
    }

    @OnClick(R.id.btn_upload)
    void upload() {
        mViewModel.upload();
    }

    @OnClick(R.id.btn_download)
    void download() {
        mViewModel.download();
    }

    @OnClick(R.id.btn_send)
    void sendMessage() {
        mViewModel.sendMessage();
    }

}
