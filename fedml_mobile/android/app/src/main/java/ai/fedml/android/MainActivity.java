package ai.fedml.android;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE};
    private static int REQUEST_PERMISSION_CODE = 1; //请求状态码


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
            }
        }

        // make_dir();
        // run initialize(String) to avoid cold boot
        // initialize("rnn");
        System.out.println("finish initialize");

        onCLick();

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION_CODE) {
            for (int i = 0; i < permissions.length; i++) {
                Log.i("MainActivity", "permission：" + permissions[i] + ", result: " + grantResults[i]);
            }
        }
    }

    private void onCLick() {
        Button button = findViewById(R.id.button_500);
        button.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                try {
                    // measure the on-device trianing time of pre-defined DNN
                    // modify the line below if needed, e.g. olaf_celeba();
                    // modify the hyper-parameter if needed
                    // note that we only measure the training time of one batch, so the optimizer, params, initializer do not matter.
                    // functions we used are olaf_*
                    // olaf_reddit();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }


}
