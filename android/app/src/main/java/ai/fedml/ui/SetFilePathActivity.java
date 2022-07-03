package ai.fedml.ui;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.File;
import java.util.ArrayList;

import ai.fedml.R;
import ai.fedml.base.AppManager;
import ai.fedml.base.BaseActivity;
import ai.fedml.edge.FedEdgeManager;
import ai.fedml.ui.adapter.FileItem;
import ai.fedml.ui.adapter.RvFilePathAdapter;
import ai.fedml.utils.FileFilter;
import ai.fedml.utils.FileOpenUtils;
import ai.fedml.utils.FormatUtils;

/**
 * SetFilePath
 */
public class SetFilePathActivity extends BaseActivity implements View.OnClickListener, RvFilePathAdapter.OnItemClickListener {

    private TextView tv_path;
    private RecyclerView rv_file_path;
    private RvFilePathAdapter rvFilePathAdapter;

    private File[] files;// get everything in the directory
    private File currentPath;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_set_file_path);
        initView();
        loadData();

    }

    private void initView() {
        Button btn_back = findViewById(R.id.btn_back);
        Button btn_save_path = findViewById(R.id.btn_save_path);
        tv_path = findViewById(R.id.tv_path);
        rv_file_path = findViewById(R.id.rv_file_path);

        btn_back.setOnClickListener(this);
        btn_save_path.setOnClickListener(this);
    }

    private void loadData() {
        ArrayList<FileItem> dataset = new ArrayList<>();
        LinearLayoutManager layoutManager = new LinearLayoutManager(this);
        rv_file_path.setLayoutManager(layoutManager);
        rvFilePathAdapter = new RvFilePathAdapter(this, dataset);
        rv_file_path.setAdapter(rvFilePathAdapter);
        rvFilePathAdapter.setOnItemClickListener(this);

        // Get the directory of the sd card
        if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())) {
            File sd = Environment.getExternalStorageDirectory();// Get the directory of the sd card
            // get the contents of the directory
            showDir(sd);
        }
    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_save_path:
                FedEdgeManager.getFedEdgeApi().setPrivatePath(currentPath.toString());
                AppManager.getAppManager().finishActivity();
                break;
            case R.id.btn_back:
                // Load parent directory ParentFile: parent directory
                File path = currentPath.getParentFile();
                if (path == null || path.toString().equals("/storage/emulated")) {
                    AppManager.getAppManager().finishActivity();
                } else {
                    showDir(path);
                }
                break;
        }
    }


    /**
     * Load all folders and files and update the interface
     */
    @SuppressLint("NotifyDataSetChanged")
    private void showDir(File dir) {
        // save current location
        currentPath = dir;
        // Get the contents of the directory (listFiles: get all the contents), and filter the files and folders starting with "." by FileFilter() function
        files = dir.listFiles(new FileFilter());
        if (files == null) {
            return;
        }
        rvFilePathAdapter.mSetData.clear();
        for (File file : files) {
            FileItem item = FileItem.builder().fileIcon(file.isFile() ? R.mipmap.ic_file : R.mipmap.ic_dir)
                    .fileName(file.getName())
                    .fileSize(FormatUtils.unitConversion(file.length()))
                    .fileLastModifiedTime(FormatUtils.longToString(file.lastModified()))
                    .build();
            rvFilePathAdapter.mSetData.add(item);
        }
        tv_path.setText(currentPath.toString());
        rvFilePathAdapter.notifyDataSetChanged();

    }


    @Override
    public void onItemClick(View view, int position) {
        if (files[position].isFile()) {
            // open the file
            FileOpenUtils.openFile(this, files[position]);
        } else {
            // Open the contents of the file directory
            // load new data
            showDir(files[position]);
        }
    }
}
