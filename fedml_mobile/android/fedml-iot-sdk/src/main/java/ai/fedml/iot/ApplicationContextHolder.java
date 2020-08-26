package ai.fedml.iot;

import android.app.Application;

/**
 * Created by hechaoyang on 3/8/18.
 */

public class ApplicationContextHolder {

    private static Application applicatonContext = null;

    public static void setContext(Application application){
        applicatonContext = application;
    }

    public static Application getContext(){
        return applicatonContext;
    }
}
