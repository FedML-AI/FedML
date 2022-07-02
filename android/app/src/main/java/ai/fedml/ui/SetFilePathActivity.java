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
 *
 * @author xkai
 * @date 2021/12/31 16:17
 */
public class SetFilePathActivity extends BaseActivity implements View.OnClickListener, RvFilePathAdapter.OnItemClickListener {

    private TextView tv_path;
    private RecyclerView rv_file_path;
    private RvFilePathAdapter rvFilePathAdapter;

    private File[] files;// 获得目录中的所有内容
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

        // 获得sd卡的目录
        if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())) {
            File sd = Environment.getExternalStorageDirectory();// 获得sd卡的目录
            // 获得目录中的内容
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
                // 加载上一级目录 ParentFile：上一级目录
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
     * 加载所有文件夹和文件，并更新在界面
     */
    @SuppressLint("NotifyDataSetChanged")
    private void showDir(File dir) {
        // 保存当前位置
        currentPath = dir;
        // 获得目录中的内容（listFiles:获得所有内容），并且过滤FileFilter()以"."开头的文件和文件夹
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
            // 打开文件
            FileOpenUtils.openFile(this, files[position]);
        } else {
            // 打开文件目录中的内容
            // 加载新的数据
            showDir(files[position]);
        }
    }
}
