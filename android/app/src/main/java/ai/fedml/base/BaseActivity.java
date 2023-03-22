package ai.fedml.base;

import android.app.Activity;
import android.os.Bundle;

import ai.fedml.utils.StatusBarUtil;
import androidx.annotation.Nullable;

public class BaseActivity extends Activity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        AppManager.getAppManager().addActivity(this);
        setStatusBar();

    }

    protected void setStatusBar() {
        //Two things are done here, so that the immersive status bar of the second case mentioned at the beginning can be realized.
        // 1.Make the status bar transparent and fill the contentView to the status bar
        // 2.Reserve the position of the status bar to prevent the controls on the interface from being too close to the top.
        StatusBarUtil.setTransparent(this);
    }





    @Override
    protected void onDestroy() {
        super.onDestroy();
        AppManager.getAppManager().removeActivity(this);
    }
}
