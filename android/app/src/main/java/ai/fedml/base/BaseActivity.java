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
        //这里做了两件事情，这样就可以实现开头说的第二种情况的沉浸式状态栏了
        // 1.使状态栏透明并使contentView填充到状态栏
        // 2.预留出状态栏的位置，防止界面上的控件离顶部靠的太近。
        StatusBarUtil.setTransparent(this);
    }





    @Override
    protected void onDestroy() {
        super.onDestroy();
        AppManager.getAppManager().removeActivity(this);
    }
}
