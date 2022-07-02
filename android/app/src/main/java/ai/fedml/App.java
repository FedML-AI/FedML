package ai.fedml;

import android.app.Application;

import ai.fedml.edge.FedEdgeManager;


public class App extends Application {


    private static App app;

    @Override
    public void onCreate() {
        super.onCreate();
        app = this;
        FedEdgeManager.getFedEdgeApi().init(this);
    }


    public static App getApp() {
        return app;
    }

    @Override
    public void onTerminate() {
        super.onTerminate();
    }
}
